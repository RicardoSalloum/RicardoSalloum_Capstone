using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;

// ==================== DATA STRUCTURES (MUST BE OUTSIDE CLASS) ====================
// These classes mirror the JSON payloads exchanged with the Python backend.
// JsonUtility requires fields to be public and names to match the JSON keys exactly.

[System.Serializable]
public class LectureSendPayload
{
    public string lecture;        // full lecture text to send to the backend
    public int student_id;
    public string attention_type;
    public bool use_blanking;   // tells backend whether to apply sentence dropout
}

[System.Serializable]
public class LectureStatsResponse
{
    public string status;
    public StudentStatistics statistics;   // mirrored from BaseStudent.get_statistics()
}

// Field names here must match BaseStudent.get_statistics() exactly.
// Previous versions had "masked_words" and "mask_rate" which silently
// deserialized to zero — renamed to "zoned_words" and "zone_rate" to fix that.
[System.Serializable]
public class StudentStatistics
{
    public int student_id;
    public string attention_type;
    public string tier;            // "A", "B", or "C" — authoritative from backend

    // Sentence-level blanking pass stats
    public int blanked_sentences;
    public int total_sentences;
    public float blank_rate;

    // Markov word masking pass stats
    public int zoned_words;
    public int total_words;
    public float zone_rate;

    // Per-episode stats from the Markov model
    public int zone_out_episodes;
    public float avg_zoneout_length;

    // Theoretical retention from the Markov math vs. actual measured retention
    public float theoretical_retention;
    public int retained_words;
    public int original_words;
    public float effective_retention;

    // Total character count of the student's stored context
    public int context_length;

    // Raw Markov profile parameters (read-only mirror of the backend config)
    public float p_dropout;
    public float p_recover;
}

[System.Serializable]
public class QuestionPayload
{
    public string studentName;
    public int studentID;
    public string question;
}

[System.Serializable]
public class AnswerResponse
{
    public string student;
    public string question;
    public string answer;
    public float confidence;
}

[System.Serializable]
public class GenerateQuestionPayload
{
    public int student_id;
    public string student_name;
}

[System.Serializable]
public class MCQPayload
{
    public int student_id;
    public string question;
    public List<string> options;
}

[System.Serializable]
public class MCQResponse
{
    public int choice_index;   // which option (0-based) the student chose
    public float confidence;
    public bool no_context;     // true means the backend had to random-guess
}


// ==================== STUDENT AGENT ====================

// One instance of this component exists per student GameObject in the scene.
// It handles all HTTP communication with the Python backend for its student.
public class StudentAgent : MonoBehaviour
{
    [Header("Identity & Parameters")]
    public string studentName;
    public int studentID;
    public string attentionType;   // "low", "medium", "high", or "perfect"
    public string tier;            // "A", "B", or "C" — synced from backend after lecture
    public string backendUrl;

    [Header("Visual Indicators")]
    public Renderer studentRenderer;
    public Color normalColor = Color.white;
    public Color questioningColor = Color.yellow;

    // Internal flag so we can track whether this student has a pending question
    private bool hasQuestion = false;

    // ── Lecture statistics (mirrored from the backend after SendLecture) ──────
    public int blankedSentences = 0;
    public int totalSentences = 0;
    public int zonedWords = 0;
    public int totalWords = 0;
    public int zoneOutEpisodes = 0;
    public float avgZoneoutLength = 0f;
    public float theoreticalRetention = 0f;
    public float effectiveRetention = 0f;

    // ==================== INITIALIZATION ====================

    // Called by ClassroomManager after Instantiating the student prefab.
    // Sets up identity fields and does a local tier assignment as a placeholder
    // until the backend returns the authoritative value.
    public void Initialize(int id, string name, string type, string url)
    {
        studentID = id;
        studentName = name;
        attentionType = type;
        backendUrl = url;
        gameObject.name = "Student_" + name;

        // Local tier assignment matches ATTENTION_PROFILES in BaseStudent.py.
        // This is overwritten with the backend's value after SendLecture completes.
        switch (type.ToLower())
        {
            case "perfect":
            case "high":
                tier = "A";
                break;

            case "medium":
                tier = "B";
                break;

            case "low":
                tier = "C";
                break;

            default:
                tier = "B";
                break;
        }

        // Try to find a Renderer on this GameObject if none was assigned in the Inspector
        if (studentRenderer == null)
        {
            studentRenderer = GetComponent<Renderer>();
        }
    }

    // ==================== LECTURE PROCESSING ====================

    // Sends the full lecture text to the backend for this student's attention model.
    // The backend runs blanking + Markov masking and returns statistics we mirror here.
    public IEnumerator SendLecture(string lectureText, bool applyBlanking = true)
    {
        string url = $"{backendUrl}/send_lecture";

        var payload = new LectureSendPayload
        {
            lecture = lectureText,
            student_id = studentID,
            attention_type = attentionType,
            use_blanking = applyBlanking
        };

        string json = JsonUtility.ToJson(payload);

        using (UnityWebRequest req = new UnityWebRequest(url, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            req.uploadHandler = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");

            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                var response = JsonUtility.FromJson<LectureStatsResponse>(req.downloadHandler.text);

                if (response.statistics != null)
                {
                    var s = response.statistics;

                    // The backend is the source of truth for tier assignment
                    if (!string.IsNullOrEmpty(s.tier))
                    {
                        tier = s.tier;
                    }

                    // Mirror all statistics into public fields for Inspector visibility
                    blankedSentences = s.blanked_sentences;
                    totalSentences = s.total_sentences;
                    zonedWords = s.zoned_words;
                    totalWords = s.total_words;
                    zoneOutEpisodes = s.zone_out_episodes;
                    avgZoneoutLength = s.avg_zoneout_length;
                    theoreticalRetention = s.theoretical_retention;
                    effectiveRetention = s.effective_retention;

                    Debug.Log(
                        $"<color=green>[{studentName}] Tier {tier} | " +
                        $"Blanked: {blankedSentences}/{totalSentences} sentences | " +
                        $"Zoned: {zonedWords}/{totalWords} words " +
                        $"({s.zone_rate * 100f:F0}%) | " +
                        $"Episodes: {zoneOutEpisodes} avg {avgZoneoutLength:F1}w | " +
                        $"Retention: {effectiveRetention * 100f:F0}% " +
                        $"(theory {theoreticalRetention * 100f:F0}%)</color>"
                    );
                }
            }
            else
            {
                Debug.LogError($"[{studentName}] Failed to send lecture: {req.error}");
            }
        }
    }

    // ==================== QUESTIONING ====================

    // Asks the backend to answer a free-text question for this student.
    // The result is delivered via the callback so the caller can display it.
    public IEnumerator AskQuestion(string question, System.Action<AnswerResponse> callback)
    {
        string url = $"{backendUrl}/ask_question";

        var payload = new QuestionPayload
        {
            studentName = studentName,
            studentID = studentID,
            question = question
        };

        string json = JsonUtility.ToJson(payload);

        using (UnityWebRequest req = new UnityWebRequest(url, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            req.uploadHandler = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");

            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                callback?.Invoke(JsonUtility.FromJson<AnswerResponse>(req.downloadHandler.text));
            }
            else
            {
                // Pass null so the caller knows the request failed
                callback?.Invoke(null);
            }
        }
    }

    // Asks the backend whether this student wants to raise their hand right now.
    // If they do, the student turns yellow and the question is delivered to the callback.
    public IEnumerator GenerateQuestion(System.Action<StudentQuestionResponse> callback)
    {
        string url = $"{backendUrl}/generate_question";

        var payload = new GenerateQuestionPayload
        {
            student_id = studentID,
            student_name = studentName
        };

        string json = JsonUtility.ToJson(payload);

        using (UnityWebRequest req = new UnityWebRequest(url, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            req.uploadHandler = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");

            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                StudentQuestionResponse resp =
                    JsonUtility.FromJson<StudentQuestionResponse>(req.downloadHandler.text);

                if (resp.has_question)
                {
                    // Student is raising their hand — update the visual and notify the caller
                    hasQuestion = true;
                    UpdateVisualState();
                    callback?.Invoke(resp);
                }
                else
                {
                    // Student doesn't have a question this cycle
                    callback?.Invoke(null);
                }
            }
        }
    }

    // ==================== MCQ TESTING ====================

    // Sends one MCQ question and its options to the backend for this student.
    // The backend runs the full RAG + LLM pipeline and returns the chosen index.
    public IEnumerator SubmitMCQ(string qText, List<string> opts,
                                  System.Action<int, float, bool> onComplete)
    {
        string url = $"{backendUrl}/answer_mcq";

        var payload = new MCQPayload
        {
            student_id = studentID,
            question = qText,
            options = opts
        };

        string json = JsonUtility.ToJson(payload);

        using (UnityWebRequest req = new UnityWebRequest(url, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            req.uploadHandler = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");

            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                MCQResponse res = JsonUtility.FromJson<MCQResponse>(req.downloadHandler.text);
                onComplete?.Invoke(res.choice_index, res.confidence, res.no_context);
            }
            else
            {
                Debug.LogError($"[{studentName}] MCQ submission failed: {req.error}");

                // Fall back to option 0 with zero confidence so scoring still proceeds
                onComplete?.Invoke(0, 0f, true);
            }
        }
    }

    // ==================== VISUAL FEEDBACK ====================

    // Applies the correct colour to this student's mesh based on whether
    // they currently have a pending question.
    void UpdateVisualState()
    {
        if (studentRenderer == null)
        {
            return;
        }

        if (hasQuestion)
        {
            studentRenderer.material.color = questioningColor;
        }
        else
        {
            studentRenderer.material.color = normalColor;
        }
    }

    // Called by ClassroomManager after the teacher answers a student's question
    // to reset the visual indicator back to the normal colour.
    public void ClearQuestionState()
    {
        hasQuestion = false;
        UpdateVisualState();
    }
}
