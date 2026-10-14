import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# =====================================================================
# TASK 1 & 2: REGEX PARSER (Handled by your teammate)
# =====================================================================
def parse_with_regex(text: str) -> dict | None:
    """
    Fast regex extractor for known standard bank alert patterns.
    Returns full dict on match, or None if regex fails.
    """
    # Sample regex logic (your teammate can extend these patterns)
    amount_match = re.search(r"(?:NGN|N|Amt:?\s*)([\d,]+(?:\.\d{2})?)", text, re.IGNORECASE)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{2}-[A-Za-z]{3}-\d{4})", text)
    time_match = re.search(r"(\d{2}:\d{2})", text)
    
    # Simple channel detection
    channel = None
    lower_text = text.lower()
    if "pos" in lower_text:
        channel = "pos"
    elif "atm" in lower_text:
        channel = "atm"
    elif "trf" in lower_text or "transfer" in lower_text:
        channel = "transfer"
    elif "web" in lower_text or "online" in lower_text:
        channel = "web"

    # Reference detection
    ref_match = re.search(r"(?:Ref|Desc|Session ID)?:?\s*(\d{10,30})", text, re.IGNORECASE)

    if amount_match and date_match and channel:
        raw_amt = amount_match.group(1).replace(",", "")
        return {
            "amount": float(raw_amt),
            "date": date_match.group(1),
            "time": time_match.group(1) if time_match else "00:00",
            "channel": channel,
            "bank": "UnknownBank",
            "reference": ref_match.group(1) if ref_match else "UNKNOWN"
        }
    return None


# =====================================================================
# TASK 3: YOUR RESPONSIBILITY - LLM FALLBACK & STRICT VALIDATION
# =====================================================================
def parse_with_llm(text: str) -> dict | None:
    """
    Fallback extractor using Groq (Llama 3.3).
    Extracts transaction fields when regex fails.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[Warning] GROQ_API_KEY not found in environment.")
        return None

    client = Groq(api_key=api_key)

    system_prompt = """You are an expert Nigerian bank transaction parser.
Extract the transaction details and return ONLY a valid JSON object.
Allowed channels: "atm", "pos", "transfer", "web".

JSON schema:
{
  "amount": number,
  "date": "YYYY-MM-DD",
  "time": "HH:MM",
  "channel": "pos" | "atm" | "transfer" | "web",
  "bank": "Bank Name",
  "reference": "reference or session number"
}
Do not include markdown tags (```json), commentary, or extra text."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Alert text:\n{text}"}
            ],
            model="qwen/qwen3.8-27b",
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        content = chat_completion.choices[0].message.content.strip()
        data = json.loads(content)
        return data

    except Exception as e:
        print(f"[LLM Error]: {e}")
        return None


def validate_extracted_data(data: dict) -> tuple[bool, list[str]]:
    """
    Task 4: Strict validation of required fields and types.
    """
    required_fields = ["amount", "date", "time", "channel", "bank", "reference"]
    missing = []

    if not isinstance(data, dict):
        return False, required_fields

    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            missing.append(field)

    # Validate amount is a number
    if "amount" not in missing:
        try:
            data["amount"] = float(data["amount"])
            if data["amount"] <= 0:
                missing.append("amount (invalid value)")
        except (ValueError, TypeError):
            missing.append("amount")

    # Validate channel is one of the 4 accepted values
    if "channel" not in missing:
        data["channel"] = str(data["channel"]).lower()
        if data["channel"] not in ["atm", "pos", "transfer", "web"]:
            missing.append("channel")

    # Format date to YYYY-MM-DD if possible
    if "date" not in missing:
        raw_date = str(data["date"])
        try:
            # Check if it matches standard date formatting
            for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y"):
                try:
                    parsed_dt = datetime.strptime(raw_date, fmt)
                    data["date"] = parsed_dt.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
        except Exception:
            pass

    return (len(missing) == 0), missing


# =====================================================================
# TASK 4: SAFE CORE FUNCTION (NEVER CRASHES)
# =====================================================================
def parse_alert(text: str) -> dict:
    """
    Primary interface for ReverseIt alert parsing:
    1. Tries Regex first (fast, free)
    2. Falls back to LLM if regex fails
    3. Strictly validates the result
    4. Guarantees safety: returns either valid fields or could_not_parse
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {"error": "could_not_parse", "missing": ["input_text_empty"]}

    parsed_result = None

    # Step 1: Try regex
    try:
        parsed_result = parse_with_regex(text)
    except Exception as e:
        print(f"[Regex Exception]: {e}")
        parsed_result = None

    # Step 2: Fallback to LLM if regex fails or missed key fields
    if not parsed_result:
        parsed_result = parse_with_llm(text)

    # Step 3: Validate output
    is_valid, missing_fields = validate_extracted_data(parsed_result)

    if is_valid:
        return {
            "amount": parsed_result["amount"],
            "date": parsed_result["date"],
            "time": str(parsed_result.get("time", "00:00")),
            "channel": parsed_result["channel"],
            "bank": str(parsed_result.get("bank", "Unknown Bank")),
            "reference": str(parsed_result.get("reference", "N/A"))
        }
    else:
        return {
            "error": "could_not_parse",
            "missing": missing_fields
        }


# =====================================================================
# TEST RUNNER (Task 5)
# =====================================================================
if __name__ == "__main__":
    print("=== Testing ReverseIt Alert Parser ===\n")

    test_alerts = [
        # Test 1: Standard alert
        "Acct: 012****789 Amt: NGN 25,000.00 Dr Desc: POS WDL / 0001293849182 / Opay Terminal Ikeja Date: 2026-10-14 14:32",
        
        # Test 2: Informal / Messy bank alert (Regex may struggle, LLM handles it)
        "Debit Alert: Your Access Bank account was debited with 15000 NGN for ATM cash withdrawal at Victoria Island on 2026-10-13 18:20. Ref: 9948271038.",
        
        # Test 3: Completely invalid text (Testing that it never crashes)
        "Hello, I sent money to my sister yesterday morning and she has not received it yet."
    ]

    for i, alert in enumerate(test_alerts, 1):
        print(f"--- Alert Test #{i} ---")
        output = parse_alert(alert)
        print(json.dumps(output, indent=2))
        print()