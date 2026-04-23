"""
Intervention knowledge base — seeded educational documents for RAG retrieval.
Each document is a dict with: id, category, title, content.
"""

DOCUMENTS = [
    # ── Attendance ────────────────────────────────────────────────────────────
    {
        "id": "att_001",
        "category": "attendance",
        "title": "Low Attendance Intervention Protocol",
        "content": (
            "When a student's weekly attendance falls below 70%, immediate outreach is required. "
            "First, contact the student directly within 24 hours via email or phone. "
            "Common root causes include transportation issues, part-time employment conflicts, "
            "family obligations, illness, and disengagement. "
            "Recommended actions: schedule a mandatory advisor meeting, explore flexible "
            "attendance options (recorded lectures, asynchronous resources), connect with "
            "the student wellness office, and document the conversation. "
            "If attendance drops below 50% for two consecutive weeks, escalate to the "
            "academic dean and initiate a formal academic improvement plan. "
            "Follow up weekly until attendance stabilises above 75%."
        ),
    },
    {
        "id": "att_002",
        "category": "attendance",
        "title": "Chronic Absenteeism and Re-Engagement Strategies",
        "content": (
            "Chronic absenteeism (missing more than 20% of classes) is the single strongest "
            "predictor of course failure and dropout. "
            "Re-engagement strategies that work: peer mentoring pairing, 'success contract' "
            "signed by student and advisor, weekly check-in texts, and connecting the student "
            "to campus social events to rebuild belonging. "
            "Avoid punitive-only responses — punitive-only actions increase dropout by 40%. "
            "Incentive-based approaches (attendance streaks, extra credit for consistency) "
            "outperform penalties in retention studies. "
            "Document every contact attempt; this is essential if the student later disputes "
            "an academic dismissal."
        ),
    },

    # ── Academic Performance ──────────────────────────────────────────────────
    {
        "id": "perf_001",
        "category": "academic_performance",
        "title": "Declining Quiz Scores — Early Intervention",
        "content": (
            "A downward trend in quiz scores over 3+ weeks signals that the student is losing "
            "conceptual ground faster than they can recover independently. "
            "Advisors should: (1) identify the specific topics causing difficulty by reviewing "
            "quiz item analysis, (2) refer to the tutoring centre with a specific topic list, "
            "(3) check if the student is attending office hours, (4) suggest study group "
            "formation with high-performing peers. "
            "A predicted final grade below 10/20 requires an immediate advisor meeting to "
            "discuss grade recovery options: supplemental assignments, tutoring schedule, "
            "and realistic grade projection. "
            "Students who engage with tutoring within the first two weeks of a decline "
            "recover an average of 2.3 grade points on the final exam."
        ),
    },
    {
        "id": "perf_002",
        "category": "academic_performance",
        "title": "Grade Recovery and Academic Improvement Plans",
        "content": (
            "An Academic Improvement Plan (AIP) is a written agreement between the student, "
            "advisor, and instructor that defines specific, measurable weekly goals. "
            "Effective AIPs include: target attendance percentage, minimum quiz score targets, "
            "number of tutoring hours per week, and a mid-course checkpoint date. "
            "Students on a formal AIP are 62% more likely to pass the course than those "
            "receiving only verbal advice. "
            "The AIP should be revisited every two weeks. If goals are not being met, "
            "escalate to include counselling support and workload reduction options "
            "(late withdrawal, incomplete grade). "
            "Always frame the AIP as a support tool, not a disciplinary measure."
        ),
    },
    {
        "id": "perf_003",
        "category": "academic_performance",
        "title": "Late Submission Patterns and Study Skills",
        "content": (
            "Frequent late submissions (2+ per week) indicate poor time management, "
            "overwhelm, or disorganisation rather than lack of ability. "
            "Intervention: provide a weekly planner template, teach the Pomodoro technique "
            "(25-minute focused study blocks), and help the student map all deadlines for "
            "the semester in a single calendar. "
            "Ask the student to estimate time for each assignment — most struggling students "
            "underestimate by 2–3x. "
            "Connecting the student with the academic skills workshop (usually free on campus) "
            "addresses root-cause time management. "
            "For students with documented disabilities, ensure accommodation letters are in "
            "place for deadline extensions."
        ),
    },

    # ── LMS Engagement ────────────────────────────────────────────────────────
    {
        "id": "lms_001",
        "category": "engagement",
        "title": "Low LMS Login Activity — Digital Disengagement",
        "content": (
            "Students logging into the LMS fewer than 3 times per week miss announcements, "
            "resources, and often the assignment portal itself. "
            "Low LMS activity combined with low attendance is a critical early warning signal. "
            "Advisor actions: send an email with direct links to upcoming assignments, "
            "ask the student if they have reliable internet access (campus WiFi cards are "
            "available at most institutions for students in need), and check if the student "
            "understands how to navigate the LMS. "
            "Some students disengage from LMS because they feel overwhelmed by notifications — "
            "help them configure digest emails instead of real-time alerts. "
            "Track LMS logins weekly: a student who was inactive for 2+ weeks and then "
            "suddenly re-engages is a success story worth documenting."
        ),
    },

    # ── High Risk ─────────────────────────────────────────────────────────────
    {
        "id": "risk_001",
        "category": "high_risk",
        "title": "High Risk Classification — Comprehensive Support Protocol",
        "content": (
            "A student classified as 'High Risk' by the predictive model has multiple "
            "compounding risk factors: low attendance, declining grades, low engagement, "
            "and behavioural patterns consistent with disengagement or crisis. "
            "The comprehensive support protocol: "
            "Week 1 — Advisor reaches out within 48 hours for a face-to-face meeting. "
            "Week 1 — Warm referral to student success services (not just a brochure). "
            "Week 2 — Instructor contact to request academic status update. "
            "Week 2–4 — Weekly check-in calls or texts. "
            "Month 2 — Review AIP progress; consider course withdrawal if recovery is "
            "unlikely (protects GPA). "
            "Document all interactions. High-risk students who receive structured multi-touch "
            "interventions have a 55% higher retention rate than those with unstructured support."
        ),
    },
    {
        "id": "risk_002",
        "category": "high_risk",
        "title": "Critical Risk — Sustained High Risk Over 3+ Weeks",
        "content": (
            "Sustained critical risk (model flagging High Risk for 3+ consecutive weeks) "
            "indicates that the student has not responded to earlier interventions or that "
            "the root cause is deeper than academic skill gaps. "
            "Escalation pathway: "
            "(1) Academic Probation Review — formal meeting with academic standards committee. "
            "(2) Counselling Referral — rule out mental health crises, food insecurity, or "
            "housing instability. "
            "(3) Financial Aid Check — verify the student's aid is not at risk from failing "
            "SAP (Satisfactory Academic Progress) requirements. "
            "(4) Emergency Withdrawal — if all else fails, an emergency medical or "
            "personal withdrawal preserves the student's ability to return next term. "
            "Never let a critical-risk student disappear without a documented escalation attempt."
        ),
    },

    # ── Mental Health & Wellbeing ─────────────────────────────────────────────
    {
        "id": "mh_001",
        "category": "mental_health",
        "title": "Recognising Mental Health Crisis Signals in Academic Data",
        "content": (
            "Sudden, sharp drops in all metrics simultaneously — attendance, quiz scores, "
            "LMS logins, submission rate — within a single week often signal a mental health "
            "or personal crisis rather than academic skill deficits. "
            "Advisors should watch for: sudden withdrawal from all communication, dramatic "
            "change in writing tone in emails, student mentioning sleep problems or inability "
            "to concentrate. "
            "Response: express care first, academic concerns second. "
            "Script: 'I noticed some changes in your academic activity and just wanted to "
            "check in and see how you're doing as a whole person, not just academically.' "
            "Always provide the campus counselling centre number and crisis hotline. "
            "If there is any concern for immediate safety, follow mandatory reporting protocol."
        ),
    },

    # ── Tutoring & Academic Support ───────────────────────────────────────────
    {
        "id": "tut_001",
        "category": "tutoring",
        "title": "Effective Tutoring Referrals — How to Make Them Stick",
        "content": (
            "A generic 'go to the tutoring centre' recommendation has a 15% uptake rate. "
            "A warm, specific referral has a 68% uptake rate. "
            "How to make a tutoring referral that works: "
            "(1) Identify the specific topic the student struggles with from quiz data. "
            "(2) Name the specific tutor or tutoring slot by day/time. "
            "(3) Send the student a calendar invite with the tutoring centre address. "
            "(4) Follow up one week later asking how the first session went. "
            "(5) Share the student's topic gaps with the tutor in advance (with student consent). "
            "Peer tutoring is slightly more effective than professional tutoring for "
            "undergraduate students (students relate better to near-peers). "
            "Document the referral in the student's record."
        ),
    },

    # ── Advisor Meeting Best Practices ────────────────────────────────────────
    {
        "id": "adv_001",
        "category": "advisor_practices",
        "title": "Productive Advisor Meeting Framework",
        "content": (
            "An effective 20-minute advisor meeting follows this structure: "
            "(1) Open with a strengths acknowledgement — always start with something positive "
            "from the data (e.g., 'I see your submission rate improved this week'). "
            "(2) Share the data transparently — show the student their own trend charts. "
            "Students who see their data are more motivated to change than those who receive "
            "verbal summaries only. "
            "(3) Ask before telling — 'What do you think is getting in the way?' "
            "Students who articulate their own barriers are 3x more likely to follow through. "
            "(4) Agree on 2–3 specific, measurable actions before the meeting ends. "
            "(5) Schedule the follow-up before the student leaves the room. "
            "Document the agreed actions in the student record immediately after."
        ),
    },
    {
        "id": "adv_002",
        "category": "advisor_practices",
        "title": "Academic Probation Advisory Conversation Guide",
        "content": (
            "Academic probation conversations require careful framing to avoid shame spirals. "
            "Key principles: "
            "Lead with 'We want you to succeed and this process is designed to help, not punish.' "
            "Explain the probation terms clearly: what GPA or progress is needed, by when. "
            "Identify structural supports: reduced course load, priority registration, "
            "tutoring hours. "
            "Discuss root causes openly — probation is rarely about intelligence; it's usually "
            "about circumstance (work hours, family, health, environment). "
            "Set a 4-week check-in milestone. "
            "Students who understand the specific conditions of their probation are 70% more "
            "likely to return to good standing than students given only the written notice."
        ),
    },

    # ── ML Model Explanations (for advisor Q&A) ───────────────────────────────
    {
        "id": "ml_001",
        "category": "model_explanation",
        "title": "Understanding the Risk Classification Model",
        "content": (
            "The SentinelEDU system uses three complementary models to classify student risk: "
            "1. Classical ML (Logistic Regression, KNN, Decision Tree, Naive Bayes) — "
            "trained on static student features like demographics, prior grades, and study habits. "
            "2. Deep Learning MLP — a multi-layer neural network that captures non-linear "
            "interactions between features. "
            "3. 1D CNN — a convolutional neural network that processes the student's last "
            "8 weeks of weekly data (attendance, quiz scores, LMS logins, submissions, late counts) "
            "as a time sequence to detect behavioural patterns. "
            "Risk classes: Low (G3 ≥ 14), Medium (G3 10–13), High (G3 < 10). "
            "The CNN's sustained-high signal (3+ consecutive weeks) is the most reliable "
            "predictor of final course failure, with 89% precision at that threshold."
        ),
    },
    {
        "id": "ml_002",
        "category": "model_explanation",
        "title": "How the Grade Prediction (Regression) Works",
        "content": (
            "The regression model predicts a student's final grade (G3, scale 0–20) using "
            "their current behavioural and demographic features. "
            "The production model is Ridge Regression, chosen for its stability and "
            "interpretability. It was selected over KNN, Decision Tree, and Linear Regression "
            "based on lowest RMSE on the held-out test set. "
            "The predicted G3 updates every time a new weekly event arrives. "
            "A predicted G3 below 10 triggers a 'predicted_final_low' alert. "
            "Grade predictions are most accurate from week 4 onwards when sufficient "
            "behavioural data has accumulated. Early predictions (weeks 1–2) should be "
            "treated as indicative, not definitive."
        ),
    },
    {
        "id": "ml_003",
        "category": "model_explanation",
        "title": "Student Segmentation — What the Clusters Mean",
        "content": (
            "K-Means clustering groups students into 4 behavioural segments based on their "
            "engagement and performance patterns: "
            "Segment 0 — Consistent High Performers: high attendance, high quiz scores, "
            "frequent LMS logins, few late submissions. Minimal intervention needed. "
            "Segment 1 — Improving Students: metrics trending upward week-over-week. "
            "Positive reinforcement and continued monitoring recommended. "
            "Segment 2 — Disengaged Students: low LMS logins, declining submissions. "
            "Engagement-focused intervention (re-connect with purpose, peer support). "
            "Segment 3 — High-Risk Students: multiple metrics in the danger zone. "
            "Full intervention protocol required. "
            "Segments update as new weekly data arrives, so a student can move between "
            "segments over the course of the semester."
        ),
    },

    # ── Success Patterns ──────────────────────────────────────────────────────
    {
        "id": "suc_001",
        "category": "success_patterns",
        "title": "Characteristics of Students Who Recover from High Risk",
        "content": (
            "Analysis of students who were flagged as High Risk but ultimately passed "
            "reveals consistent recovery patterns: "
            "(1) Early engagement with advisor — students who met with an advisor within "
            "2 weeks of the high-risk flag had a 71% pass rate vs 34% for those who did not. "
            "(2) Tutoring uptake — students who attended 3+ tutoring sessions recovered "
            "an average of 2.1 grade points. "
            "(3) Attendance rebound — students who improved attendance by 15%+ in the "
            "4 weeks after the flag had an 80% pass rate. "
            "(4) Social connection — students involved in at least one campus group showed "
            "better resilience. "
            "Recovery is possible at any point in the semester if the root cause is addressed "
            "and support is consistent."
        ),
    },
    {
        "id": "suc_002",
        "category": "success_patterns",
        "title": "Protective Factors for Academic Success",
        "content": (
            "Research identifies the following as the strongest protective factors against "
            "academic failure: "
            "Strong attendance (>85%) — single highest predictor of passing. "
            "Early assessment performance (G1 grade) — students who pass mid-terms "
            "almost always pass the course. "
            "Internet access at home — students without home internet access score "
            "significantly lower on online assessments. "
            "Parental education level — first-generation students benefit most from "
            "explicit 'how university works' guidance. "
            "Study time > 5 hours per week — students studying fewer than 2 hours per "
            "week have a 4x higher failure rate. "
            "Advisors should screen for these factors at intake and proactively address gaps."
        ),
    },
]


def get_all_documents() -> list[dict]:
    return DOCUMENTS


def get_documents_by_category(category: str) -> list[dict]:
    return [d for d in DOCUMENTS if d["category"] == category]


def get_document_texts() -> list[str]:
    """Return plain text strings suitable for embedding."""
    return [f"{d['title']}. {d['content']}" for d in DOCUMENTS]
