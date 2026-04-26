using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Questioning system with two modes toggled by useLLMQuestions:
///
///   TRUE  — sends the recent lecture snippet to the LLM backend
///           (/generate_lecture_question) and speaks whatever it returns.
///
///   FALSE — picks a question from the test bank loaded via Initialize(),
///           using keyword matching against the recent snippet (contextual)
///           with a random fallback.
/// </summary>
public class QuestioningManager : MonoBehaviour
{
    // References to other components this script depends on
    [Header("References")]
    public ClassroomManager classroomManager; // manages the classroom scene and students
    public AudioSource audioSource;           // plays the TTS audio

    // The local TTS server address and which voice to use
    [Header("TTS Settings")]
    public string ttsBaseUrl = "http://127.0.0.1:5003";
    public string voice = "af_bella";

    // The local AI agent server that generates questions from lecture content
    [Header("Agent Backend")]
    public string agentBaseUrl = "http://127.0.0.1:5002";

    // Toggle between LLM-generated questions and pre-written test bank questions
    [Header("Question Mode")]
    [Tooltip("TRUE  = LLM generates a question from the live lecture snippet.\nFALSE = pick a question from the loaded test bank.")]
    public bool useLLMQuestions = true;

    // Controls how often questions are asked and when to start
    [Header("Timing")]
    public float questionCooldown = 60f;      // seconds between questions after the first
    [Tooltip("Delay before the very first question (seconds)")]
    public float firstQuestionDelay = 120f;   // wait this long before asking the first question

    // Internal list of exam questions used in test bank mode
    private List<ExamQuestion> questionPool;

    // Tracks time until next question is asked
    private float cooldownTimer;

    // Flag that prevents Update() from doing anything until Initialize() is called
    private bool isReady = false;

    // ==================== LIFECYCLE ====================

    void Start()
    {
        // If no AudioSource was assigned in the Inspector, create one at runtime with safe defaults
        if (audioSource == null)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
            audioSource.playOnAwake = false;  // don't play anything automatically
            audioSource.spatialBlend = 0f;    // 0 = fully 2D audio (not positional)
            audioSource.volume = 1.0f;
        }

        // Start the countdown using the first-question delay, not the regular cooldown
        cooldownTimer = firstQuestionDelay;

        // Log the current mode and server URLs so we can confirm settings at startup
        Debug.Log($"[QuestioningManager] Mode: {(useLLMQuestions ? "LLM" : "Test Bank")}");
        Debug.Log($"[QuestioningManager] TTS   URL: {ttsBaseUrl}/synthesize");
        Debug.Log($"[QuestioningManager] Agent URL: {agentBaseUrl}/generate_lecture_question");
        Debug.Log($"<color=cyan>[QuestioningManager] First question in {firstQuestionDelay}s ({firstQuestionDelay / 60f:F1} min)</color>");
    }

    void Update()
    {
        // Don't do anything until the manager has been initialized
        if (!isReady)
        {
            return;
        }

        // In test-bank mode, we can't ask questions if the pool is empty or missing
        if (!useLLMQuestions && (questionPool == null || questionPool.Count == 0))
        {
            return;
        }

        // Count down the timer each frame
        cooldownTimer -= Time.deltaTime;

        // When the timer hits zero, reset it and trigger a question
        if (cooldownTimer <= 0f)
        {
            cooldownTimer = questionCooldown;
            StartCoroutine(AskQuestion());
        }
    }

    // ==================== PUBLIC API ====================

    // Overload for LLM mode — no question pool needed, just flip the ready flag
    public void Initialize()
    {
        questionPool = null;
        isReady = true;
        Debug.Log("<color=green>[QuestioningManager] Ready — LLM mode</color>");
    }

    // Overload for test bank mode — stores the question list and enables the system
    public void Initialize(List<ExamQuestion> questions)
    {
        questionPool = questions;
        isReady = true;

        // Log different messages depending on which mode is actually active
        if (useLLMQuestions)
        {
            Debug.Log($"<color=green>[QuestioningManager] Ready — LLM mode (test bank of {questions.Count} stored but not used for questioning)</color>");
        }
        else
        {
            Debug.Log($"<color=green>[QuestioningManager] Ready — Test Bank mode ({questions.Count} questions)</color>");
        }
    }

    // Halts all questioning and stops any audio that's currently playing
    public void Stop()
    {
        isReady = false;
        StopAllCoroutines(); // cancel any in-progress question or TTS coroutine

        if (audioSource != null && audioSource.isPlaying)
        {
            audioSource.Stop();
            Debug.Log("<color=red>[QuestioningManager] Audio stopped</color>");
        }

        Debug.Log("<color=red>[QuestioningManager] Stopped</color>");
    }

    // Re-enables the system after a Stop() and resets the cooldown timer
    public void Resume()
    {
        isReady = true;
        cooldownTimer = questionCooldown;
        Debug.Log("<color=green>[QuestioningManager] Resumed</color>");
    }

    // ==================== CORE FLOW ====================

    // Entry point for asking a question — picks a random student then routes to the right mode
    IEnumerator AskQuestion()
    {
        var students = classroomManager.GetStudents();

        // Skip this cycle if there are no students in the classroom
        if (students == null || students.Count == 0)
        {
            yield break;
        }

        // Pick a random student to be the one asking the question
        StudentAgent student = students[Random.Range(0, students.Count)];

        // Delegate to the appropriate questioning method based on the mode toggle
        if (useLLMQuestions)
        {
            yield return StartCoroutine(AskLLMQuestion(student));
        }
        else
        {
            yield return StartCoroutine(AskTestBankQuestion(student));
        }
    }

    // ==================== LLM MODE ====================

    // Grabs the current lecture snippet, sends it to the AI, and speaks the returned question
    IEnumerator AskLLMQuestion(StudentAgent student)
    {
        // Get whatever the professor has said recently
        string snippet = classroomManager.GetCurrentLectureSnippet();

        // Can't generate a question without any lecture content to base it on
        if (string.IsNullOrEmpty(snippet))
        {
            Debug.LogWarning("[QuestioningManager] No lecture snippet yet — skipping LLM question.");
            yield break;
        }

        // Variable that will be populated by the callback inside FetchLLMQuestion
        string generatedQuestion = null;
        yield return StartCoroutine(FetchLLMQuestion(snippet, q => generatedQuestion = q));

        // If the server came back empty or failed, skip this round
        if (string.IsNullOrEmpty(generatedQuestion))
        {
            Debug.LogWarning("[QuestioningManager] LLM returned no question — skipping.");
            yield break;
        }

        Debug.Log($"<color=yellow>[Question - LLM] {student.studentName} asks: {generatedQuestion}</color>");

        // Have the student speak the question aloud via TTS
        yield return StartCoroutine(SpeakText($"Excuse me professor, {generatedQuestion}"));
    }

    // Simple wrapper class needed because JsonUtility only serializes class fields, not raw strings
    [System.Serializable]
    private class LLMQuestionRequest { public string snippet; }

    // Makes a POST request to the agent backend, passing the lecture snippet and waiting for a question
    IEnumerator FetchLLMQuestion(string snippet, System.Action<string> callback)
    {
        string url = $"{agentBaseUrl}/generate_lecture_question";

        // Wrap the snippet in a serializable object so JsonUtility produces valid JSON
        string json = JsonUtility.ToJson(new LLMQuestionRequest { snippet = snippet });

        using (UnityWebRequest req = new UnityWebRequest(url, "POST"))
        {
            // Convert the JSON string to raw bytes and set it as the request body
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            req.uploadHandler = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerBuffer(); // stores the response in memory
            req.SetRequestHeader("Content-Type", "application/json");

            // Pause here until the web request completes
            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                // Parse the JSON response into our response model
                var resp = JsonUtility.FromJson<LLMQuestionResponse>(req.downloadHandler.text);

                // Only pass the question back if the server actually returned one
                if (resp.has_question)
                {
                    callback(resp.question);
                }
                else
                {
                    callback(null);
                }
            }
            else
            {
                Debug.LogError($"[QuestioningManager] LLM fetch failed: {req.error}");
                callback(null);
            }
        }
    }

    // Maps to the JSON shape the agent backend returns: { "has_question": bool, "question": string }
    [System.Serializable]
    private class LLMQuestionResponse
    {
        public bool has_question; // true if the LLM decided a question was appropriate
        public string question;     // the actual question text if has_question is true
    }

    // ==================== TEST BANK MODE ====================

    // Tries to find a contextually relevant question, then falls back to a random one
    IEnumerator AskTestBankQuestion(StudentAgent student)
    {
        string snippet = classroomManager.GetCurrentLectureSnippet();

        ExamQuestion question = null;
        bool isContextual = false;

        // Only attempt keyword matching if there is a lecture snippet to compare against
        if (!string.IsNullOrEmpty(snippet))
        {
            question = PickContextualQuestion(snippet);
            isContextual = (question != null); // mark whether we got a topic-relevant match
        }

        // If contextual matching found nothing, fall back to a completely random question
        if (question == null)
        {
            question = questionPool[Random.Range(0, questionPool.Count)];
            isContextual = false;
        }

        // Label the log entry so we can tell at a glance how the question was selected
        string label = isContextual ? "CONTEXTUAL" : "RANDOM";
        Debug.Log($"<color=yellow>[Question - {label}] {student.studentName} asked: {question.text}</color>");

        yield return StartCoroutine(SpeakText($"Excuse me professor, I have a question: {question.text}"));
    }

    // Scores every question in the pool by how many lecture keywords it contains, returns the best match
    ExamQuestion PickContextualQuestion(string lectureSnippet)
    {
        // Nothing to match against if the snippet or pool is missing
        if (string.IsNullOrEmpty(lectureSnippet) || questionPool == null || questionPool.Count == 0)
        {
            return null;
        }

        // Tokenize the lecture text into individual lowercase words
        string[] lectureWords = lectureSnippet.ToLower()
            .Split(new[] { ' ', '.', ',', '!', '?', '\n', '\t' },
                   System.StringSplitOptions.RemoveEmptyEntries);

        // Common words that don't help identify topic — we'll exclude these from matching
        HashSet<string> stopWords = new HashSet<string>
        {
            "the","a","an","and","or","but","in","on","at","to","for","of","with","by","from",
            "is","are","was","were","be","been","have","has","had","do","does","did","will",
            "would","could","should","may","might","can","this","that","these","those",
            "i","you","he","she","it","we","they","what","which","who","when","where","why","how"
        };

        // Keep only meaningful words — longer than 3 chars and not in the stop list
        List<string> keywords = lectureWords
            .Where(w => w.Length > 3 && !stopWords.Contains(w))
            .Distinct()
            .ToList();

        // If all words were filtered out, there's nothing useful to match on
        if (keywords.Count == 0)
        {
            return null;
        }

        // Score each question by counting how many keywords appear in its text
        var scored = new List<(ExamQuestion q, int score)>();

        foreach (var q in questionPool)
        {
            string qt = q.text.ToLower();
            int hits = keywords.Count(kw => qt.Contains(kw));

            // Only include questions that matched at least one keyword
            if (hits > 0)
            {
                scored.Add((q, hits));
            }
        }

        // If nothing matched, signal the caller to use a random fallback instead
        if (scored.Count == 0)
        {
            return null;
        }

        // Sort best matches to the front
        scored = scored.OrderByDescending(x => x.score).ToList();

        // Pick randomly from the top 3 so we don't always repeat the same best-matching question
        int topCount = Mathf.Min(3, scored.Count);
        var pick = scored[Random.Range(0, topCount)];

        Debug.Log($"<color=cyan>[Contextual Match] Score: {pick.score}</color>");
        return pick.q;
    }

    // ==================== TTS ====================

    // Sends text to the TTS server, receives WAV audio back, and plays it through the AudioSource
    IEnumerator SpeakText(string text)
    {
        string url = $"{ttsBaseUrl}/synthesize";

        // Build the request body with the text and chosen voice
        TTSRequest payload = new TTSRequest { text = text, voice = voice };
        string json = JsonUtility.ToJson(payload);

        Debug.Log($"[TTS] POST {url}");
        Debug.Log($"[TTS] Body: {json}");

        using (UnityWebRequest req = new UnityWebRequest(url, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            req.uploadHandler = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerBuffer(); // holds the raw WAV bytes in memory
            req.SetRequestHeader("Content-Type", "application/json");

            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                // Convert the raw WAV bytes into a Unity AudioClip
                AudioClip clip = WavToAudioClip(req.downloadHandler.data);

                if (clip != null)
                {
                    audioSource.clip = clip;
                    audioSource.Play();

                    // Log just the first 50 characters of the text so the log stays readable
                    Debug.Log($"[TTS] Playing: {text.Substring(0, Mathf.Min(50, text.Length))}...");

                    // Wait the full duration of the clip before continuing
                    yield return new WaitForSeconds(clip.length);
                }
                else
                {
                    Debug.LogError("[TTS] Failed to create AudioClip from WAV data");
                }
            }
            else
            {
                Debug.LogError($"[TTS] Error: {req.error}");
            }
        }
    }

    // ==================== WAV PARSER ====================

    // Manually reads the WAV header and sample data, then builds a Unity AudioClip from it
    AudioClip WavToAudioClip(byte[] wavData)
    {
        // A valid WAV file must be at least 44 bytes (the standard header size)
        if (wavData.Length < 44)
        {
            return null;
        }

        // Read audio format details from their fixed positions in the WAV header
        int channels = System.BitConverter.ToInt16(wavData, 22); // mono = 1, stereo = 2
        int sampleRate = System.BitConverter.ToInt32(wavData, 24); // e.g. 22050, 44100
        int bitsPerSample = System.BitConverter.ToInt16(wavData, 34); // typically 16

        // Audio data starts right after the 44-byte header
        int dataIndex = 44;
        int dataSize = wavData.Length - dataIndex;       // total bytes of audio data
        int bytesPerSamp = bitsPerSample / 8;                // 16-bit = 2 bytes per sample
        int totalSamples = dataSize / bytesPerSamp / channels; // number of samples per channel

        // Unity AudioClip needs normalized float samples in the range [-1, 1]
        float[] samples = new float[totalSamples * channels];

        for (int i = 0; i < totalSamples * channels; i++)
        {
            int byteIndex = dataIndex + (i * bytesPerSamp);

            if (bitsPerSample == 16)
            {
                // Read a 16-bit signed integer and normalize it to the float range
                short s16 = System.BitConverter.ToInt16(wavData, byteIndex);
                samples[i] = s16 / 32768f; // 32768 is the max value of a signed 16-bit int
            }
        }

        // Create the AudioClip and load the normalized sample data into it
        AudioClip clip = AudioClip.Create("TTS", totalSamples, channels, sampleRate, false);
        clip.SetData(samples, 0);
        return clip;
    }
}

// Data class used to serialize the TTS request body as JSON
[System.Serializable]
public class TTSRequest
{
    public string text;  // the sentence or phrase to be spoken
    public string voice; // the voice ID to use (e.g. "af_bella")
}
