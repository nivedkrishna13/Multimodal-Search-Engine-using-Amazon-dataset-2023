import re

def parse_query(query):
    filters = {}

    match = re.search(r'under \$?(\d+)', query.lower())
    if match:
        filters["max_price"] = float(match.group(1))

    match = re.search(r'above (\d(\.\d)?)', query.lower())
    if match:
        filters["min_stars"] = float(match.group(1))

    if "cheaper" in query.lower():
        filters["sort_by"] = "price_asc"

    if "best rated" in query.lower():
        filters["sort_by"] = "stars_desc"

    return filters