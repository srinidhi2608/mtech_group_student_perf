import pandas as pd

def load_students_csv(f):
    """
    Loads students CSV into a DataFrame.
    Expects at least an id/name column. Returns DataFrame.
    """
    df = pd.read_csv(f)
    return df

def load_predictions_csv(f):
    """
    Loads predictions CSV. Expects a column like 'student_id' and 'predicted_ctc'.
    """
    df = pd.read_csv(f)
    return df

def get_student_record(student_df, identifier):
    """
    Try to fetch student by common id/name columns.
    identifier is a string representation from the selectbox.
    """
    if student_df is None:
        return None
    # Try matching common columns
    for col in ["student_id", "id", "studentID", "roll", "name"]:
        if col in student_df.columns:
            # match as string
            matches = student_df[student_df[col].astype(str) == str(identifier)]
            if not matches.empty:
                return matches.iloc[0].to_dict()
    # fallback: try first column
    first_col = student_df.columns[0]
    matches = student_df[student_df[first_col].astype(str) == str(identifier)]
    if not matches.empty:
        return matches.iloc[0].to_dict()
    return None

def get_predicted_ctc(pred_df, identifier):
    if pred_df is None:
        return None
    # common id names + predicted column
    id_cols = [c for c in pred_df.columns if c.lower() in ("student_id","id","studentid","roll","name")]
    pred_cols = [c for c in pred_df.columns if "pred" in c.lower() and "ctc" in c.lower() or c.lower().startswith("predicted")]
    # fallback pred column selection
    if not pred_cols:
        for c in pred_df.columns:
            if "ctc" in c.lower():
                pred_cols.append(c)
    if not pred_cols:
        return None
    pred_col = pred_cols[0]
    # match identifier
    if id_cols:
        id_col = id_cols[0]
        matches = pred_df[pred_df[id_col].astype(str) == str(identifier)]
        if not matches.empty:
            return matches.iloc[0][pred_col]
    # no id found: if the CSV has a single row, return that
    if len(pred_df) == 1:
        return pred_df.iloc[0][pred_col]
    return None