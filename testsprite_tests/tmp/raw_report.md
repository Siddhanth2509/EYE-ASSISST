
# TestSprite AI Testing Report(MCP)

---

## 1️⃣ Document Metadata
- **Project Name:** EYE-ASSISST
- **Date:** 2026-04-27
- **Prepared by:** TestSprite AI Team

---

## 2️⃣ Requirement Validation Summary

#### Test TC001 Block access to dashboard when not authenticated
- **Test Code:** [TC001_Block_access_to_dashboard_when_not_authenticated.py](./TC001_Block_access_to_dashboard_when_not_authenticated.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/fe3a7fc0-7dc5-4b79-a98a-a073e03040a5
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC002 Log in as Patient and see patient dashboard
- **Test Code:** [TC002_Log_in_as_Patient_and_see_patient_dashboard.py](./TC002_Log_in_as_Patient_and_see_patient_dashboard.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/afa90877-5ab9-4feb-b7c8-31bafb2aee76
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC003 Persist logged-in session across refresh
- **Test Code:** [TC003_Persist_logged_in_session_across_refresh.py](./TC003_Persist_logged_in_session_across_refresh.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/39f22946-adf9-4997-bbcd-b348482aa0db
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC004 Doctor can log in and access the dashboard
- **Test Code:** [TC004_Doctor_can_log_in_and_access_the_dashboard.py](./TC004_Doctor_can_log_in_and_access_the_dashboard.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/a1b7d6ae-955d-45f6-bb8a-1d332751dd10
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC005 Technician can log in and access the dashboard
- **Test Code:** [TC005_Technician_can_log_in_and_access_the_dashboard.py](./TC005_Technician_can_log_in_and_access_the_dashboard.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/0152fffb-f41f-4347-bbba-c16b82e1502d
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC006 Admin can log in and access the dashboard
- **Test Code:** [TC006_Admin_can_log_in_and_access_the_dashboard.py](./TC006_Admin_can_log_in_and_access_the_dashboard.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/4a882179-8e7d-473f-9a92-4da54930856c
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC007 Log in as Doctor and see doctor dashboard
- **Test Code:** [TC007_Log_in_as_Doctor_and_see_doctor_dashboard.py](./TC007_Log_in_as_Doctor_and_see_doctor_dashboard.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/36c7adff-7c4e-42c2-97e0-313677ed08ea
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC008 Sign up as a new user and land on the correct role dashboard
- **Test Code:** [TC008_Sign_up_as_a_new_user_and_land_on_the_correct_role_dashboard.py](./TC008_Sign_up_as_a_new_user_and_land_on_the_correct_role_dashboard.py)
- **Test Error:** TEST BLOCKED

The signup and login pages did not load, so the signup flow could not be exercised.

Observations:
- Navigated to /login but the page is blank and contains 0 interactive elements.
- The SPA appears not to have rendered; no login or sign-up UI is visible.
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/261e026d-5b17-4c7b-a6fd-721b2aa923ab
- **Status:** BLOCKED
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC009 Technician can submit a scan for AI analysis and view results
- **Test Code:** [TC009_Technician_can_submit_a_scan_for_AI_analysis_and_view_results.py](./TC009_Technician_can_submit_a_scan_for_AI_analysis_and_view_results.py)
- **Test Error:** TEST BLOCKED

The required fundus image file is not available to the browser, so the image upload and analysis steps cannot be completed.

Observations:
- The file upload control (input index 930) is present on the page but the test file path /tmp/fundus.jpg is not available to the agent.
- Without a file to upload, the 'Analyze Image' action (button index 963) cannot be performed and AI results cannot be verified.
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/a13b0b1f-7e3b-46af-b580-58b76c9b44b6
- **Status:** BLOCKED
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC010 Doctor can review a pending scan with overlay toggle and submit a decision
- **Test Code:** [TC010_Doctor_can_review_a_pending_scan_with_overlay_toggle_and_submit_a_decision.py](./TC010_Doctor_can_review_a_pending_scan_with_overlay_toggle_and_submit_a_decision.py)
- **Test Error:** TEST FAILURE

Submitting the doctor's review did not work — the app returned an error and the scan stayed in the pending list.

Observations:
- After clicking 'Submit Review' a red notification appeared: 'Failed to submit review: Failed to submit review'.
- TEST-SCAN-001 is still present in the Pending Reviews list.
- No success confirmation or removal from the pending list was observed.
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/84199b56-b97c-4ca2-a761-148791945b5c
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC011 Doctor can modify severity and notes when choosing Modify
- **Test Code:** [TC011_Doctor_can_modify_severity_and_notes_when_choosing_Modify.py](./TC011_Doctor_can_modify_severity_and_notes_when_choosing_Modify.py)
- **Test Error:** TEST FAILURE

Changing the diagnosis severity is not possible because the UI does not expose a severity control after selecting 'Modify diagnosis'.

Observations:
- The 'Modify diagnosis' radio was selectable and the clinical notes textarea appeared.
- There is no visible dropdown, selector, or input for changing the severity value on the review form.
- The page shows a 'Submit Review' button, but without a way to set the new severity the requested change cannot be recorded.
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/2627c769-83d5-4fe6-8984-ae973a369b94
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC012 Navigate from landing page to role-based login
- **Test Code:** [TC012_Navigate_from_landing_page_to_role_based_login.py](./TC012_Navigate_from_landing_page_to_role_based_login.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/72f4aefd-39ae-4870-8a01-9207ff04b74b
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC013 Admin can add a user and see it in the user list
- **Test Code:** [TC013_Admin_can_add_a_user_and_see_it_in_the_user_list.py](./TC013_Admin_can_add_a_user_and_see_it_in_the_user_list.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/33b82b97-2f65-45e2-aa00-d420372d0dd3
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC014 View scan details from scan history
- **Test Code:** [TC014_View_scan_details_from_scan_history.py](./TC014_View_scan_details_from_scan_history.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/4f062cf7-c490-4a08-a32e-0ced12424f12
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC015 Book an appointment and see confirmation
- **Test Code:** [TC015_Book_an_appointment_and_see_confirmation.py](./TC015_Book_an_appointment_and_see_confirmation.py)
- **Test Error:** TEST FAILURE

The booking could not be completed — the payment step did not finalize and no booking confirmation appeared in the patient portal.

Observations:
- After clicking 'Bypass Payment (Test Automation)' and attempting UPI payment, the payment modal remained visible.
- No booking confirmation, toast, or new upcoming appointment was shown in the patient portal.
- 'Bypass Payment' was clicked 3 times and 'Pay ₹500 via UPI' was clicked 1 time with no effect.
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/24081de4-b6d0-4189-ac9e-0b915a94ef58/f0c8ca3c-acec-43bd-9517-963ef5e1a846
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---


## 3️⃣ Coverage & Matching Metrics

- **66.67** of tests passed

| Requirement        | Total Tests | ✅ Passed | ❌ Failed  |
|--------------------|-------------|-----------|------------|
| ...                | ...         | ...       | ...        |
---


## 4️⃣ Key Gaps / Risks
{AI_GNERATED_KET_GAPS_AND_RISKS}
---