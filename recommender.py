from typing import List, Dict
from syllabus_parser import build_keyword_index
from data_processing import get_student_record, get_predicted_ctc

# Simple skill-to-uplift mapping (example). In real system this should be learned from data.
SKILL_UPLIFT = {
    "machine learning": 0.12,
    "deep learning": 0.15,
    "data structures": 0.08,
    "algorithms": 0.1,
    "system design": 0.13,
    "cloud": 0.11,
    "devops": 0.07,
    "nlp": 0.12,
    "computer vision": 0.12,
    "sql": 0.05,
    "python": 0.03,
    "communication": 0.02
}

# Keywords to search in syllabus to map to skills
SKILL_KEYWORDS = {
    "machine learning": ["machine learning", "ml", "supervised", "unsupervised"],
    "deep learning": ["deep learning", "neural network", "cnn", "rnn", "transformer"],
    "data structures": ["data structure", "linked list", "tree", "graph", "heap"],
    "algorithms": ["algorithm", "sorting", "searching", "dynamic programming"],
    "system design": ["system design", "scalability", "microservices"],
    "cloud": ["aws", "azure", "gcp", "cloud"],
    "devops": ["docker", "kubernetes", "ci/cd", "devops"],
    "nlp": ["nlp", "natural language", "text processing"],
    "computer vision": ["computer vision", "image processing", "opencv"],
    "sql": ["sql", "database", "relational database"],
    "python": ["python programming", "python"],
    "communication": ["communication", "soft skill", "presentation"]
}

def _keywords_to_skills(found_keywords: List[str]) -> List[str]:
    """
    Map matched syllabus keywords back to skills (reverse mapping)
    """
    found_skills = set()
    lower_found = [k.lower() for k in found_keywords]
    for skill, kws in SKILL_KEYWORDS.items():
        for kw in kws:
            if any(kw in ff for ff in lower_found):
                found_skills.add(skill)
    return list(found_skills)

def suggest_courses_for_student(student_df, pred_df, student_identifier, syllabus_texts: Dict[str, str]) -> List[Dict]:
    """
    Suggest courses from provided syllabus_texts based on student's record and predicted CTC.
    Returns a list of dicts: {course_title, matched_skills, estimated_uplift}
    Implementation: keyword matching of syllabus text -> skills -> estimated uplift using mapping.
    """
    if not syllabus_texts:
        return []

    # Build a keyword index (we will check for skill keywords presence)
    all_keywords = [kw for kws in SKILL_KEYWORDS.values() for kw in kws]
    index = build_keyword_index(syllabus_texts, all_keywords)

    # Flatten possible course titles and matches
    candidates = []
    for fname, matched in index.items():
        if not matched:
            continue
        # Map matched keywords to skills
        matched_skills = _keywords_to_skills(matched)
        # Compute estimated uplift as sum of skill uplifts (simple proxy)
        uplift = 0.0
        for sk in matched_skills:
            uplift += SKILL_UPLIFT.get(sk, 0.02)
        candidates.append({
            "course_title": fname,
            "matched_skills": matched_skills,
            "estimated_uplift": f"{uplift*100:.1f}%"
        })

    # Sort by uplift desc
    candidates.sort(key=lambda x: float(x["estimated_uplift"].strip('%')), reverse=True)

    # Optionally personalize with student's predicted CTC severity
    if student_identifier and pred_df is not None:
        pred_ctc = get_predicted_ctc(pred_df, student_identifier)
        # If predicted CTC is low, be more aggressive (no real change to ranking here, but could filter)
        if pred_ctc is not None:
            # example: if pred_ctc < some threshold, return top 6 else top 4
            threshold = 5.0  # 5 LPA
            cap = 6 if pred_ctc < threshold else 4
            return candidates[:cap]
    # default
    return candidates[:5]