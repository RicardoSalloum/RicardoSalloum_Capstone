using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using System.IO;

// ClassroomManager is the central controller for the entire VR classroom simulation.
// It spawns students, sends them lectures, runs the exam phase, and exports the report.
public class ClassroomManager : MonoBehaviour
{
    [Header("Student Grid / Prefab")]
    public GameObject studentPrefab;
    // Total student count = sum of all four attention counts below — no separate field needed
    public int columns = 3;
    public float spacing = 2f;

    [Header("Character Appearance Randomization")]
    [Tooltip("4 male character prefabs (the person mesh only)")]
    public GameObject[] malePrefabs = new GameObject[4];
    [Tooltip("4 female character prefabs (the person mesh only)")]
    public GameObject[] femalePrefabs = new GameObject[4];
    [Tooltip("Probability 0-1 that a student is male (0.5 = 50/50)")]
    [Range(0f, 1f)]
    public float maleSpawnChance = 0.5f;

    [Tooltip("Hair color materials to randomly assign to the Hair renderer.")]
    public Material[] hairColorMaterials;
    [Tooltip("Eye color materials to randomly assign to the Face renderer's eye slot.")]
    public Material[] eyeColorMaterials;
    [Tooltip("Which material slot on the Face SkinnedMeshRenderer is the eye color.")]
    public int eyeMaterialIndex = 3;
    [Tooltip("Name of the child GameObject to replace with the character prefab (e.g. 'Capsule').")]
    public string characterSlotChildName = "Capsule";

    [Header("Research Distribution")]
    public int lowAttentionCount = 3;
    public int mediumAttentionCount = 3;
    public int highAttentionCount = 2;
    public int perfectAttentionCount = 1;

    [Header("Real-Time Questioning")]
    public bool enableRealTimeQuestions = true;
    public float questionCheckInterval = 10f;
    public int maxQuestionsPerMinute = 3;

    [Header("TTS Audio")]
    public AudioSource studentQuestionAudioSource;
    public float audioFetchTimeout = 10f;

    [Header("Lecture Input")]
    [Tooltip("Optional: drag an MP3/WAV file here to use a pre-recorded lecture")]
    public AudioClip preRecordedLecture;

    [Header("Live Lecture (Microphone)")]
    [TextArea(3, 10)]
    public string loadedLectureText;   // accumulates as snippets arrive from VoiceController

    [Header("Current Lecture Snippet")]
    [TextArea(2, 5)]
    [Tooltip("Most recent lecture snippet (last N words of speech)")]
    public string currentLectureSnippet = "";

    [Header("Lecture Snippet Settings")]
    [Tooltip("Maximum words to keep in the rolling snippet window")]
    public int maxSnippetWords = 100;

    [Header("Question (Live QA)")]
    [TextArea(1, 3)]
    public string questionToAsk = "What is photosynthesis?";

    [Header("Testing Phase - Defaults")]
    [Tooltip("Fallback exam file path if PlayerPrefs is not set")]
    public string examDataFallbackPath = "C:/ClassroomData/exam.json";
    [Tooltip("Fallback test bank path if PlayerPrefs is not set")]
    public string testBankFallbackPath = "C:/ClassroomData/testbank.json";

    [Header("Backend URLs")]
    public string agentUrl = "http://127.0.0.1:5002";
    public string ttsUrl = "http://127.0.0.1:5003";

    [Header("Statistics & Visualization")]
    public bool showStatistics = true;

    // All spawned StudentAgent components — used throughout the lifecycle
    private List<StudentAgent> students = new List<StudentAgent>();

    // Tracks the last time we polled students for real-time questions
    private float lastQuestionCheck = 0f;

    [Header("Questioning System")]
    public QuestioningManager questioningManager;

    // Stores exam scores keyed by tier ("A", "B", "C") for the final report
    private Dictionary<string, List<int>> tierScores = new Dictionary<string, List<int>>();

    // Chronological list of questions asked during the session (for export)
    private List<QuestionEvent> questionTimeline = new List<QuestionEvent>();

    // Tracks how many students answered each question correctly (key = question index)
    private Dictionary<int, int> questionCorrectCounts = new Dictionary<int, int>();

    // Kept so the report can access question text after the exam phase
    private ExamSheet lastExam;

    [Header("Exam Progress Bar (assign a UI Slider)")]
    [Tooltip("Optional UI Slider (0-1) that shows overall exam completion")]
    public UnityEngine.UI.Slider examProgressSlider;
    [Tooltip("Optional Text label such as '3 / 9 students done'")]
    public UnityEngine.UI.Text examProgressLabel;

    // Progress counters updated as each student answers each question
    private int _examTotalQuestions = 0;   // total = students × questions per exam
    private int _examDoneQuestions = 0;   // how many individual answers have been submitted

    // Limits how many MCQ API calls run at the same time to avoid overwhelming the backend
    private const int MAX_CONCURRENT_MCQ = 1;
    private int _activeMcqCount = 0;

    // Set to true while the lecture is live. Gates real-time questioning and TTS
    // so nothing fires after the teacher toggles the lecture off.
    private bool _lectureActive = false;

    // ─────────────────────────────────────────────────────────────────────────
    // HELPER: Returns the exam file path from PlayerPrefs, or the fallback
    // ─────────────────────────────────────────────────────────────────────────
    string GetExamFilePath()
    {
        return PlayerPrefs.GetString(PathSettingsMenu.PREF_EXAM_FILE, examDataFallbackPath);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // HELPER: Returns the test bank file path from PlayerPrefs, or the fallback
    // ─────────────────────────────────────────────────────────────────────────
    string GetTestBankFilePath()
    {
        return PlayerPrefs.GetString(PathSettingsMenu.PREF_TESTBANK_FILE, testBankFallbackPath);
    }

    // Exposes the student list to other scripts (e.g. UI panels)
    public List<StudentAgent> GetStudents()
    {
        return students;
    }

    void Start()
    {
        Debug.Log($"<color=cyan>[ClassroomManager] Exam path:     {GetExamFilePath()}</color>");
        Debug.Log($"<color=cyan>[ClassroomManager] TestBank path: {GetTestBankFilePath()}</color>");

        SpawnStudents();

        // Create an AudioSource for playing TTS answers if one wasn't assigned in the Inspector
        if (studentQuestionAudioSource == null)
        {
            GameObject audioObj = new GameObject("StudentQuestionAudio");
            audioObj.transform.SetParent(transform);
            studentQuestionAudioSource = audioObj.AddComponent<AudioSource>();
            studentQuestionAudioSource.playOnAwake = false;
            Debug.Log("<color=yellow>Created AudioSource for student questions</color>");
        }

        // If a pre-recorded lecture clip was dragged in, process it now instead of live mic
        if (preRecordedLecture != null)
        {
            Debug.Log("<color=cyan>Pre-recorded lecture detected. Processing audio...</color>");
            StartCoroutine(ProcessPreRecordedLecture());
        }
        else
        {
            Debug.Log("<color=yellow>No pre-recorded lecture. Use VoiceController (Press V) to record live.</color>");
        }

        // Load the test bank for the real-time questioning system if it exists
        string testBankPath = GetTestBankFilePath();
        if (enableRealTimeQuestions && File.Exists(testBankPath))
        {
            // InitializeQuestioningSystem(); // commented out — now started by VoiceController
        }
        else if (enableRealTimeQuestions)
        {
            Debug.LogWarning($"<color=orange>[ClassroomManager] Test bank file not found: {testBankPath}</color>");
        }
    }

    void Update()
    {
        float currentTime = Time.time;

        // Periodically check if any students want to raise their hand
        if (enableRealTimeQuestions && _lectureActive && currentTime - lastQuestionCheck >= questionCheckInterval)
        {
            StartCoroutine(CheckForStudentQuestions());
            lastQuestionCheck = currentTime;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SPAWNING
    // ─────────────────────────────────────────────────────────────────────────

    // Creates all student GameObjects in a grid layout and assigns attention types.
    // The four attention counts are combined into one shuffled list so tiers are
    // distributed randomly around the classroom rather than clustered by type.
    void SpawnStudents()
    {
        if (studentPrefab == null)
        {
            return;
        }

        students.Clear();

        // Build a flat list containing each attention type the correct number of times
        List<string> attentionProfiles = new List<string>();
        for (int i = 0; i < lowAttentionCount; i++) { attentionProfiles.Add("low"); }
        for (int i = 0; i < mediumAttentionCount; i++) { attentionProfiles.Add("medium"); }
        for (int i = 0; i < highAttentionCount; i++) { attentionProfiles.Add("high"); }
        for (int i = 0; i < perfectAttentionCount; i++) { attentionProfiles.Add("perfect"); }

        // Fisher-Yates shuffle so seat positions are randomised across attention types
        for (int i = attentionProfiles.Count - 1; i > 0; i--)
        {
            int j = Random.Range(0, i + 1);
            string tmp = attentionProfiles[i];
            attentionProfiles[i] = attentionProfiles[j];
            attentionProfiles[j] = tmp;
        }

        int totalStudents = attentionProfiles.Count;

        int cLow = attentionProfiles.Count(p => p == "low");
        int cMed = attentionProfiles.Count(p => p == "medium");
        int cHigh = attentionProfiles.Count(p => p == "high");
        int cPerfect = attentionProfiles.Count(p => p == "perfect");

        Debug.Log($"<color=cyan>[SpawnStudents] Total: {totalStudents} | " +
                  $"low={cLow}(C)  medium={cMed}(B)  high={cHigh}(A)  perfect={cPerfect}(A)</color>");

        for (int i = 0; i < totalStudents; i++)
        {
            // Calculate grid position: row and column from the linear index
            int row = i / columns;
            int col = i % columns;
            Vector3 pos = new Vector3(col * spacing, 0f, row * spacing) + transform.position;

            GameObject obj = Instantiate(studentPrefab, pos, Quaternion.identity, transform);
            StudentAgent agent = obj.GetComponent<StudentAgent>();

            if (agent != null)
            {
                agent.Initialize(i, "Student_" + (i + 1), attentionProfiles[i], agentUrl);
                students.Add(agent);
            }

            ApplyCharacterAppearance(obj);
        }

        Debug.Log($"<color=cyan>Classroom spawned with {students.Count} students.</color>");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CHARACTER APPEARANCE RANDOMIZATION
    // ─────────────────────────────────────────────────────────────────────────

    // Replaces the placeholder mesh on each student prefab with a randomly
    // chosen character model, then randomises hair and eye colours.
    void ApplyCharacterAppearance(GameObject studentRoot)
    {
        // Find the slot child that holds the placeholder mesh
        Transform slotChild = studentRoot.transform.Find(characterSlotChildName);
        if (slotChild == null)
        {
            Debug.LogWarning($"[ClassroomManager] Could not find child '{characterSlotChildName}' on {studentRoot.name}. " +
                             $"Character mesh will not be randomized.");
            return;
        }

        // Decide male or female based on the configured probability
        bool isMale = Random.value <= maleSpawnChance;
        GameObject[] pool = isMale ? malePrefabs : femalePrefabs;

        if (pool == null || pool.Length == 0)
        {
            Debug.LogWarning($"[ClassroomManager] No {(isMale ? "male" : "female")} prefabs assigned. Skipping character swap.");
            return;
        }

        // Filter out any null entries left in the array
        var validPool = System.Array.FindAll(pool, p => p != null);
        if (validPool.Length == 0)
        {
            Debug.LogWarning($"[ClassroomManager] All {(isMale ? "male" : "female")} prefab slots are empty.");
            return;
        }

        GameObject chosenPrefab = validPool[Random.Range(0, validPool.Length)];

        // Save the slot's local transform before destroying it so the replacement
        // spawns in exactly the same position, rotation, and scale
        Vector3 slotLocalPos = slotChild.localPosition;
        Quaternion slotLocalRot = slotChild.localRotation;
        Vector3 slotLocalScale = slotChild.localScale;

        Object.Destroy(slotChild.gameObject);

        // Instantiate the chosen character model inside the student root
        GameObject characterInstance = Instantiate(chosenPrefab, studentRoot.transform);
        characterInstance.name = characterSlotChildName;
        characterInstance.transform.localPosition = slotLocalPos;
        characterInstance.transform.localRotation = slotLocalRot;
        characterInstance.transform.localScale = slotLocalScale;

        // ── Hair colour randomization ──────────────────────────────────────
        if (hairColorMaterials != null && hairColorMaterials.Length > 0)
        {
            var validHair = System.Array.FindAll(hairColorMaterials, m => m != null);
            if (validHair.Length > 0)
            {
                Material chosenHair = validHair[Random.Range(0, validHair.Length)];
                Transform hairTransform = characterInstance.transform.Find("Hair");

                if (hairTransform != null)
                {
                    Renderer hairRenderer = hairTransform.GetComponent<Renderer>();
                    if (hairRenderer != null)
                    {
                        hairRenderer.material = chosenHair;
                    }
                    else
                    {
                        Debug.LogWarning($"[ClassroomManager] No Renderer on '{chosenPrefab.name}/Hair'.");
                    }
                }
                else
                {
                    Debug.LogWarning($"[ClassroomManager] No child named 'Hair' found on '{chosenPrefab.name}'.");
                }
            }
        }

        // ── Eye colour randomization ───────────────────────────────────────
        if (eyeColorMaterials != null && eyeColorMaterials.Length > 0)
        {
            var validEye = System.Array.FindAll(eyeColorMaterials, m => m != null);
            if (validEye.Length > 0)
            {
                Material chosenEye = validEye[Random.Range(0, validEye.Length)];
                Transform faceTransform = characterInstance.transform.Find("Face");

                if (faceTransform != null)
                {
                    Renderer faceRenderer = faceTransform.GetComponent<Renderer>();
                    if (faceRenderer != null)
                    {
                        // We only replace the eye material slot, leaving skin/other slots untouched
                        if (eyeMaterialIndex >= 0 && eyeMaterialIndex < faceRenderer.materials.Length)
                        {
                            Material[] mats = faceRenderer.materials;
                            mats[eyeMaterialIndex] = chosenEye;
                            faceRenderer.materials = mats;
                        }
                        else
                        {
                            Debug.LogWarning($"[ClassroomManager] eyeMaterialIndex ({eyeMaterialIndex}) out of range — " +
                                             $"'{chosenPrefab.name}/Face' has {faceRenderer.materials.Length} material(s).");
                        }
                    }
                    else
                    {
                        Debug.LogWarning($"[ClassroomManager] No Renderer on '{chosenPrefab.name}/Face'.");
                    }
                }
                else
                {
                    Debug.LogWarning($"[ClassroomManager] No child named 'Face' found on '{chosenPrefab.name}'.");
                }
            }
        }

        Debug.Log($"<color=magenta>[Appearance] {studentRoot.name} → {(isMale ? "male" : "female")} {chosenPrefab.name}</color>");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LECTURE DISTRIBUTION
    // ─────────────────────────────────────────────────────────────────────────

    // Sends the current lecture text to all students simultaneously.
    // Each student's attention model runs on the backend independently.
    public void SendLectureToAll()
    {
        if (string.IsNullOrEmpty(loadedLectureText))
        {
            Debug.LogError("No lecture text available!");
            return;
        }

        UpdateCurrentSnippet(loadedLectureText);

        foreach (var s in students)
        {
            StartCoroutine(s.SendLecture(loadedLectureText, applyBlanking: true));
        }

        Debug.Log("<color=green>Lecture sent to all students with masking/blanking applied.</color>");
    }

    // Updates the rolling snippet window that the QuestioningManager reads from.
    // Called by VoiceController after every snippet upload and by SendLectureToAll.
    public void UpdateCurrentSnippet(string fullLecture)
    {
        if (string.IsNullOrEmpty(fullLecture))
        {
            currentLectureSnippet = "";
            return;
        }

        // Split the full lecture into individual words
        string[] words = fullLecture.Split(
            new[] { ' ', '\n', '\t' },
            System.StringSplitOptions.RemoveEmptyEntries
        );

        // If the lecture is short enough, keep it all; otherwise take the last N words
        if (words.Length <= maxSnippetWords)
        {
            currentLectureSnippet = fullLecture;
        }
        else
        {
            string[] lastWords = new string[maxSnippetWords];
            System.Array.Copy(words, words.Length - maxSnippetWords, lastWords, 0, maxSnippetWords);
            currentLectureSnippet = string.Join(" ", lastWords);
        }

        Debug.Log($"<color=cyan>[Snippet] Updated — {words.Length} total words, " +
                  $"showing last {Mathf.Min(words.Length, maxSnippetWords)}</color>");
    }

    // Returns the current rolling snippet for other systems to read
    public string GetCurrentLectureSnippet()
    {
        return currentLectureSnippet;
    }

    // Wipes the loaded lecture text (e.g. before starting a new session)
    public void ClearLectureText()
    {
        loadedLectureText = "";
        Debug.Log("<color=yellow>Lecture text cleared.</color>");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // REAL-TIME QUESTIONING
    // ─────────────────────────────────────────────────────────────────────────

    // Picks up to 3 random students and asks each one if they have a question.
    // This runs every questionCheckInterval seconds while the lecture is active.
    IEnumerator CheckForStudentQuestions()
    {
        int studentsToCheck = Mathf.Min(3, students.Count);
        var randomStudents = students.OrderBy(x => Random.value).Take(studentsToCheck);

        foreach (var student in randomStudents)
        {
            yield return StartCoroutine(student.GenerateQuestion((question) =>
            {
                if (question != null)
                {
                    HandleStudentQuestion(question);
                }
            }));

            // Small delay between students to avoid hammering the backend
            yield return new WaitForSeconds(0.5f);
        }
    }

    // Logs the question event and triggers the teacher answer + TTS pipeline
    void HandleStudentQuestion(StudentQuestionResponse question)
    {
        Debug.Log($"<color=cyan>╔══════════════════════════════════════════════════╗</color>");
        Debug.Log($"<color=cyan>║ STUDENT QUESTION</color>");
        Debug.Log($"<color=cyan>╚══════════════════════════════════════════════════╝</color>");
        Debug.Log($"<color=yellow>Student: {question.student_name}</color>");
        Debug.Log($"<color=yellow>Question: {question.question}</color>");
        Debug.Log($"<color=yellow>Type: {question.question_type} | Confidence: {question.confidence:F2}</color>");

        // Record the event so it appears in the exported session report
        questionTimeline.Add(new QuestionEvent
        {
            timestamp = Time.time,
            studentName = question.student_name,
            question = question.question,
            questionType = question.question_type
        });

        StartCoroutine(AnswerStudentQuestionWithTTS(question));
    }

    // Sends the student's question to the backend teacher agent and plays
    // the answer as TTS audio through the classroom's AudioSource.
    IEnumerator AnswerStudentQuestionWithTTS(StudentQuestionResponse question)
    {
        string answerUrl = $"{agentUrl}/answer_student_question";

        // Find the student's index so the backend can retrieve their specific context
        var payload = new
        {
            question = question.question,
            student_id = System.Array.FindIndex(
                students.ToArray(), s => s.studentName == question.student_name
            )
        };

        string json = JsonUtility.ToJson(payload);

        using (UnityWebRequest req = new UnityWebRequest(answerUrl, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            req.uploadHandler = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");

            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"<color=red>Failed to get answer: {req.error}</color>");
                yield break;
            }

            var response = JsonUtility.FromJson<TeacherAnswerResponse>(req.downloadHandler.text);

            if (response == null || response.status != "success")
            {
                Debug.LogWarning($"<color=orange>Answer generation failed</color>");
                yield break;
            }

            Debug.Log($"<color=green>Answer: {response.answer}</color>");
            Debug.Log($"<color=green>Confidence: {response.confidence:F2} | Time: {response.processing_time:F2}s</color>");

            // Don't play TTS if the lecture was toggled off while we were waiting
            if (!_lectureActive)
            {
                Debug.Log("<color=grey>[TTS] Discarded — lecture already stopped.</color>");
                yield break;
            }

            // Play the answer as spoken audio through the TTS service
            yield return StartCoroutine(PlayTTSAnswer(response.answer));
        }
    }

    // Posts text to the TTS service and plays the returned WAV clip
    IEnumerator PlayTTSAnswer(string text)
    {
        string url = $"{ttsUrl}/synthesize";

        var ttsPayload = new { text = text };
        string json = JsonUtility.ToJson(ttsPayload);

        using (UnityWebRequest req = new UnityWebRequest(url, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            req.uploadHandler = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerAudioClip(url, AudioType.WAV);
            req.SetRequestHeader("Content-Type", "application/json");
            req.timeout = (int)audioFetchTimeout;

            Debug.Log($"<color=cyan>[TTS] Generating audio for answer...</color>");

            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                AudioClip clip = DownloadHandlerAudioClip.GetContent(req);

                if (clip != null && studentQuestionAudioSource != null)
                {
                    studentQuestionAudioSource.clip = clip;
                    studentQuestionAudioSource.Play();
                    Debug.Log($"<color=green>[TTS] ♫ Playing audio ({clip.length:F1}s)</color>");

                    // Wait for the clip to finish before proceeding
                    yield return new WaitForSeconds(clip.length);
                }
                else
                {
                    Debug.LogWarning("<color=orange>[TTS] Audio clip is null</color>");
                }
            }
            else
            {
                Debug.LogWarning($"<color=orange>[TTS] Failed: {req.error}</color>");
                Debug.LogWarning($"<color=orange>[TTS] Make sure TTS service is running on {ttsUrl}</color>");
            }
        }
    }

    // Asks the currently loaded question to all students and logs every response
    public void AskAllStudents()
    {
        StartCoroutine(BroadcastQuestion(questionToAsk));
    }

    // Sends the same question to every student in a staggered fashion
    IEnumerator BroadcastQuestion(string question)
    {
        foreach (var s in students)
        {
            StartCoroutine(s.AskQuestion(question, (resp) =>
            {
                if (resp != null)
                {
                    Debug.Log($"<b>[{resp.student}]</b>: {resp.answer} (Conf: {resp.confidence:F2})");
                }
            }));

            // Stagger requests slightly to avoid flooding the backend
            yield return new WaitForSeconds(0.1f);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FILE READING HELPER
    // ─────────────────────────────────────────────────────────────────────────

    // Reads a file and returns its contents, or null if the path is invalid
    string ReadFileText(string path)
    {
        if (string.IsNullOrEmpty(path))
        {
            Debug.LogError("[ClassroomManager] File path is empty.");
            return null;
        }
        if (!File.Exists(path))
        {
            Debug.LogError($"[ClassroomManager] File not found: {path}");
            return null;
        }
        return File.ReadAllText(path);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DEBUG & TESTING
    // ─────────────────────────────────────────────────────────────────────────

    // Loads and logs the exam file so you can verify it parsed correctly in the Editor
    public void DebugExamData()
    {
        string examPath = GetExamFilePath();
        string dataAsJson = ReadFileText(examPath);

        if (dataAsJson == null)
        {
            return;
        }

        Debug.Log($"<color=cyan>File: {examPath}</color>");
        Debug.Log($"<color=cyan>JSON length: {dataAsJson.Length} characters</color>");

        ExamSheet exam = JsonUtility.FromJson<ExamSheet>(dataAsJson);

        if (exam == null)
        {
            Debug.LogError("Failed to parse ExamSheet!");
            return;
        }
        if (exam.questions == null)
        {
            Debug.LogError("exam.questions is NULL!");
            return;
        }

        Debug.Log($"<color=green>✓ Loaded {exam.questions.Count} questions</color>");

        for (int i = 0; i < exam.questions.Count; i++)
        {
            var q = exam.questions[i];
            Debug.Log($"  Q{i + 1}: {q.text}");
            Debug.Log($"    Options: {q.options?.Count ?? 0}");
            Debug.Log($"    Correct: {q.correctIndex}");
        }
    }

    // Public entry point for the exam phase — called by a UI button
    public void TriggerTestingPhase()
    {
        StartCoroutine(TriggerTestingPhaseRoutine());
    }

    // Waits for any in-flight Whisper upload to settle, sends the lecture to all
    // student backends, then starts the exam.  This prevents the race where the
    // final transcription overwrites loadedLectureText AFTER SendLecture already ran.
    IEnumerator TriggerTestingPhaseRoutine()
    {
        // ── Step 1: wait for the voice pipeline to go quiet ──────────────────
        VoiceController vc = FindObjectOfType<VoiceController>();
        if (vc != null && vc.IsUploadInFlight)
        {
            Debug.Log("<color=yellow>[Exam] Waiting for voice upload to finish...</color>");
            yield return new WaitUntil(() => !vc.IsUploadInFlight);
            Debug.Log("<color=green>[Exam] Voice upload settled.</color>");
        }

        // ── Step 2: read and validate the exam file ───────────────────────────
        string examPath = GetExamFilePath();
        string dataAsJson = ReadFileText(examPath);

        if (dataAsJson == null)
        {
            Debug.LogError("<color=red>❌ Cannot start testing phase: exam file could not be read.</color>");
            Debug.LogError($"<color=red>   Path: {examPath}</color>");
            Debug.LogError("<color=red>   Set the correct path in the Main Menu settings.</color>");
            yield break;
        }

        // ── Step 3: push the lecture to every student backend ─────────────────
        if (!string.IsNullOrEmpty(loadedLectureText))
        {
            Debug.Log("<color=cyan>[Exam] Sending lecture to all students before exam...</color>");

            // Fire all SendLecture coroutines and wait for each to complete
            // before the exam starts, so backends have context to answer MCQs.
            bool allSent = false;
            int pendingSends = students.Count;

            if (pendingSends == 0)
            {
                allSent = true;
            }

            foreach (var s in students)
            {
                StartCoroutine(SendLectureAndSignal(s, loadedLectureText, () =>
                {
                    pendingSends--;
                    if (pendingSends <= 0) allSent = true;
                }));
            }

            yield return new WaitUntil(() => allSent);
            Debug.Log("<color=green>[Exam] All students received the lecture.</color>");
        }
        else
        {
            Debug.LogWarning("<color=orange>[Exam] No lecture text available — students will answer from empty context.</color>");
        }

        // ── Step 4: start the exam ────────────────────────────────────────────
        StartTestingPhaseFromAsset(dataAsJson);
    }

    // Thin wrapper around StudentAgent.SendLecture that fires a callback when done
    IEnumerator SendLectureAndSignal(StudentAgent student, string lectureText, System.Action onDone)
    {
        yield return StartCoroutine(student.SendLecture(lectureText, applyBlanking: true));
        onDone?.Invoke();
    }

    // Called by VoiceController when the lecture is toggled off.
    // Clears the live-lecture flag so Update stops polling for student questions
    // and any in-flight TTS pipeline aborts before playing audio.
    public void StopQuestioningSystem()
    {
        _lectureActive = false;

        if (questioningManager != null)
        {
            questioningManager.Stop();
        }

        // Stop any audio that is currently playing from a student question answer
        if (studentQuestionAudioSource != null && studentQuestionAudioSource.isPlaying)
        {
            studentQuestionAudioSource.Stop();
        }

        Debug.Log("<color=yellow>[ClassroomManager] Questioning and TTS stopped.</color>");
    }

    // Parses the exam JSON, initializes the questioning system with exam questions,
    // and kicks off parallel MCQ testing for all students.
    void StartTestingPhaseFromAsset(string dataAsJson)
    {
        string examPath = GetExamFilePath();
        Debug.Log($"<color=cyan>Reading exam from: {examPath}</color>");

        ExamSheet exam = JsonUtility.FromJson<ExamSheet>(dataAsJson);

        if (exam == null)
        {
            Debug.LogError("<color=red>Failed to parse ExamSheet - JSON may be invalid</color>");
            return;
        }

        if (exam.questions == null || exam.questions.Count == 0)
        {
            Debug.LogError("<color=red>exam.questions is NULL or empty - check JSON structure</color>");
            return;
        }

        Debug.Log($"<color=cyan>╔══════════════════════════════════════════════════╗</color>");
        Debug.Log($"<color=cyan>║  EXAM: {exam.questions.Count} Questions × {students.Count} Students</color>");
        Debug.Log($"<color=cyan>╚══════════════════════════════════════════════════╝</color>");

        // Give the questioning manager the exam question pool
        if (questioningManager != null)
        {
            questioningManager.Initialize(exam.questions);
            Debug.Log("<color=green>✓ Questioning system initialized</color>");
        }

        // Reset per-exam tracking before starting
        tierScores.Clear();
        questionCorrectCounts.Clear();
        lastExam = exam;

        StartCoroutine(TestAllStudentsParallel(exam));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // QUESTIONING SYSTEM STARTUP
    // Called by VoiceController when the lecture begins.
    // If useLLMQuestions is true, no test bank is needed (LLM generates questions).
    // Otherwise, loads questions from the exam file.
    // ─────────────────────────────────────────────────────────────────────────
    public void StartQuestioningSystem()
    {
        _lectureActive = true;
        if (questioningManager == null)
        {
            Debug.LogError("[ClassroomManager] QuestioningManager not assigned!");
            return;
        }

        if (questioningManager.useLLMQuestions)
        {
            // LLM mode: the system generates questions from the live lecture snippet
            questioningManager.Initialize();
            Debug.Log("<color=green>[ClassroomManager] ✓ Questioning system started in LLM mode</color>");
        }
        else
        {
            // Test bank mode: load the exam file and pre-load the question pool
            string examPath = GetExamFilePath();
            string dataAsJson = ReadFileText(examPath);

            if (dataAsJson == null)
            {
                Debug.LogError("[ClassroomManager] Cannot start questioning — exam file not found.");
                return;
            }

            ExamSheet exam = JsonUtility.FromJson<ExamSheet>(dataAsJson);

            if (exam != null && exam.questions != null && exam.questions.Count > 0)
            {
                questioningManager.Initialize(exam.questions);
                Debug.Log($"<color=green>[ClassroomManager] ✓ Questioning system started in Test Bank mode ({exam.questions.Count} questions)</color>");
            }
            else
            {
                Debug.LogError("[ClassroomManager] Exam file parsed but contained no questions.");
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PARALLEL EXAM TESTING
    // ─────────────────────────────────────────────────────────────────────────

    // Starts one TestSingleStudent coroutine per student so they all answer
    // questions concurrently, then calls GenerateComprehensiveReport when all finish.
    IEnumerator TestAllStudentsParallel(ExamSheet exam)
    {
        int totalStudents = students.Count;
        int completedStudents = 0;
        System.DateTime startTime = System.DateTime.Now;

        Debug.Log($"\n<color=yellow>⚡ PARALLEL MODE: Testing all {totalStudents} students simultaneously</color>");
        Debug.Log($"<color=yellow>   Each student will answer {exam.questions.Count} questions</color>");
        Debug.Log($"<color=yellow>   Max concurrent API calls: {MAX_CONCURRENT_MCQ}</color>\n");

        // Set up the progress bar for this exam run
        _examTotalQuestions = totalStudents * exam.questions.Count;
        _examDoneQuestions = 0;
        UpdateExamProgress(0, totalStudents, 0);
        _activeMcqCount = 0;

        foreach (var student in students)
        {
            // Each student runs independently; the lambda captures the local counter
            StartCoroutine(TestSingleStudent(student, exam, () =>
            {
                completedStudents++;
                float progress = (completedStudents / (float)totalStudents) * 100f;
                Debug.Log($"<color=cyan>[Progress] {completedStudents}/{totalStudents} students completed ({progress:F0}%)</color>");
                UpdateExamProgress(_examDoneQuestions, totalStudents, completedStudents);

                // Once all students finish, show 100% and print the report
                if (completedStudents == totalStudents)
                {
                    UpdateExamProgress(_examTotalQuestions, totalStudents, totalStudents);
                    System.TimeSpan elapsed = System.DateTime.Now - startTime;
                    Debug.Log($"\n<color=green>✓ ALL STUDENTS COMPLETED in {elapsed.TotalSeconds:F1}s</color>\n");
                    GenerateComprehensiveReport();
                }
            }));

            // Small stagger to prevent all students hitting the backend simultaneously
            yield return new WaitForSeconds(0.1f);
        }
    }

    // Runs one student through all exam questions sequentially,
    // scoring their answers and recording results per-tier.
    IEnumerator TestSingleStudent(StudentAgent student, ExamSheet exam, System.Action onComplete)
    {
        int studentScore = 0;
        int questionsAnswered = 0;
        System.DateTime studentStartTime = System.DateTime.Now;

        Debug.Log($"<color=yellow>▶ [{student.studentName}] Starting {exam.questions.Count} questions (Tier {student.tier})</color>");

        for (int qIdx = 0; qIdx < exam.questions.Count; qIdx++)
        {
            var q = exam.questions[qIdx];

            bool questionCompleted = false;
            int chosenAnswer = -1;
            float answerConfidence = 0f;

            // Wait for a free API slot before sending this request
            while (_activeMcqCount >= MAX_CONCURRENT_MCQ)
            {
                yield return new WaitForSeconds(0.1f);
            }

            _activeMcqCount++;

            yield return StartCoroutine(student.SubmitMCQ(q.text, q.options, (choice, conf, noCtx) =>
            {
                // If confidence is very low, treat the answer as a random guess
                if (conf < 0.25f)
                {
                    chosenAnswer = Random.Range(0, q.options.Count);
                }
                else
                {
                    chosenAnswer = choice;
                }

                answerConfidence = conf;
                questionCompleted = true;
            }));

            _activeMcqCount--;

            // Track overall exam progress for the progress bar
            _examDoneQuestions++;
            UpdateExamProgress(_examDoneQuestions, students.Count, -1);

            questionsAnswered++;
            bool isCorrect = (chosenAnswer == q.correctIndex);

            if (isCorrect)
            {
                studentScore++;

                // Record how many students answered this specific question correctly
                if (!questionCorrectCounts.ContainsKey(qIdx))
                {
                    questionCorrectCounts[qIdx] = 0;
                }
                questionCorrectCounts[qIdx]++;
            }

            // Log every wrong answer and every other correct one to reduce console spam
            if ((qIdx + 1) % 2 == 0 || !isCorrect || qIdx == exam.questions.Count - 1)
            {
                string mark = isCorrect ? "✓" : "✗";
                string color = isCorrect ? "green" : "red";
                Debug.Log($"  <color={color}>[{student.studentName}] Q{qIdx + 1}/{exam.questions.Count}: {mark} " +
                          $"(Score: {studentScore}/{questionsAnswered})</color>");
            }

            yield return new WaitForSeconds(0.2f);
        }

        // Store this student's total score under their tier for the summary report
        string tier = student.tier;
        if (!tierScores.ContainsKey(tier))
        {
            tierScores[tier] = new List<int>();
        }
        tierScores[tier].Add(studentScore);

        System.TimeSpan studentElapsed = System.DateTime.Now - studentStartTime;
        float percentage = (studentScore / (float)exam.questions.Count) * 100f;

        string gradeColor;
        if (percentage >= 80)
        {
            gradeColor = "green";
        }
        else if (percentage >= 60)
        {
            gradeColor = "yellow";
        }
        else
        {
            gradeColor = "red";
        }

        Debug.Log($"<color={gradeColor}>■ [{student.studentName}] FINISHED: " +
                  $"{studentScore}/{exam.questions.Count} ({percentage:F0}%) " +
                  $"in {studentElapsed.TotalSeconds:F1}s</color>");

        onComplete?.Invoke();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EXAM PROGRESS BAR
    // ─────────────────────────────────────────────────────────────────────────

    // Updates the optional UI slider and label with the current exam progress.
    // completedStudents = -1 means we're updating the answer count mid-student.
    void UpdateExamProgress(int doneQuestions, int totalStudents, int completedStudents)
    {
        if (_examTotalQuestions <= 0)
        {
            return;
        }

        float ratio = Mathf.Clamp01(doneQuestions / (float)_examTotalQuestions);

        if (examProgressSlider != null)
        {
            examProgressSlider.value = ratio;
        }

        if (examProgressLabel != null)
        {
            if (completedStudents >= 0)
            {
                // Student-level view: how many students have fully finished
                examProgressLabel.text = $"{completedStudents}/{totalStudents} students done  ({ratio * 100f:F0}%)";
            }
            else
            {
                // Question-level view: how many individual answers have been submitted
                examProgressLabel.text = $"{doneQuestions}/{_examTotalQuestions} answers  ({ratio * 100f:F0}%)";
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // REPORT GENERATION
    // ─────────────────────────────────────────────────────────────────────────

    // Prints the full tier breakdown and per-question summary to the Console,
    // then auto-exports the results to disk.
    void GenerateComprehensiveReport()
    {
        Debug.Log("\n" + new string('=', 60));
        Debug.Log("<color=cyan><b>=== COMPREHENSIVE EVALUATION REPORT ===</b></color>");
        Debug.Log(new string('=', 60));

        Debug.Log("\n<b>TIER PERFORMANCE:</b>");
        Debug.Log("Tier\tCount\tMean\tMedian\tStdDev");
        Debug.Log("--------------------------------------------------");

        float overallMean = 0f;
        int totalStudents = 0;

        foreach (var entry in tierScores.OrderBy(x => x.Key))
        {
            string tier = entry.Key;
            List<int> scores = entry.Value;

            if (scores.Count == 0)
            {
                continue;
            }

            float mean = (float)scores.Average();
            scores.Sort();
            float median = scores[scores.Count / 2];
            float variance = (float)scores.Select(s => Mathf.Pow(s - mean, 2)).Average();
            float stdDev = Mathf.Sqrt(variance);

            Debug.Log($"{tier}\t{scores.Count}\t{mean:F1}\t{median:F1}\t{stdDev:F1}");

            overallMean += mean * scores.Count;
            totalStudents += scores.Count;
        }

        if (totalStudents > 0)
        {
            overallMean /= totalStudents;
            Debug.Log("--------------------------------------------------");
            Debug.Log($"<b>OVERALL MEAN: {overallMean:F1}</b>");
        }

        // Per-question breakdown — shows which questions were hardest
        if (lastExam != null && lastExam.questions != null && questionCorrectCounts.Count > 0)
        {
            Debug.Log(new string('=', 60));
            Debug.Log("<color=cyan><b>=== PER-QUESTION SCORES ===</b></color>");
            Debug.Log($"{"Q#",-4} {"Correct",-10} {"Total",-8} {"Pass%",-8}  Question");
            Debug.Log("--------------------------------------------------");

            for (int i = 0; i < lastExam.questions.Count; i++)
            {
                int correct = questionCorrectCounts.ContainsKey(i) ? questionCorrectCounts[i] : 0;
                float pct = (correct / (float)totalStudents) * 100f;

                string qColor;
                if (pct >= 70)
                {
                    qColor = "green";
                }
                else if (pct >= 40)
                {
                    qColor = "yellow";
                }
                else
                {
                    qColor = "red";
                }

                // Truncate the question text so it fits on one log line
                string shortText = lastExam.questions[i].text.Length > 50
                    ? lastExam.questions[i].text.Substring(0, 47) + "..."
                    : lastExam.questions[i].text;

                Debug.Log($"<color={qColor}>Q{i + 1,-3} {correct,-10} {totalStudents,-8} {pct,-7:F0}%  {shortText}</color>");
            }

            Debug.Log("--------------------------------------------------");
        }

        Debug.Log(new string('=', 60) + "\n");
        AutoExportResults();
    }

    // Automatically exports results to the app's persistent data folder with a timestamp
    void AutoExportResults()
    {
        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        string fileName = $"VRClassroom_Results_{timestamp}.json";
        string folderPath = System.IO.Path.Combine(Application.persistentDataPath, "Results");

        if (!System.IO.Directory.Exists(folderPath))
        {
            System.IO.Directory.CreateDirectory(folderPath);
        }

        ExportSessionData(System.IO.Path.Combine(folderPath, fileName));
    }

    // Manual export available from the right-click context menu in the Inspector
    [ContextMenu("Export Results Now")]
    public void ManualExportResults()
    {
        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        string fileName = $"VRClassroom_Results_{timestamp}.json";
        string folderPath = System.IO.Path.Combine(Application.persistentDataPath, "Results");

        if (!System.IO.Directory.Exists(folderPath))
        {
            System.IO.Directory.CreateDirectory(folderPath);
        }

        string filePath = System.IO.Path.Combine(folderPath, fileName);
        ExportSessionData(filePath);
        Debug.Log($"<color=green>Manual export completed: {filePath}</color>");
    }

    // Sends the test bank to the backend's questioning system via HTTP
    void InitializeQuestioningSystem()
    {
        string url = $"{agentUrl}/initialize_questioning";
        string testBankPath = GetTestBankFilePath();
        string json = ReadFileText(testBankPath);

        if (json == null)
        {
            Debug.LogWarning($"<color=orange>[ClassroomManager] Could not read test bank: {testBankPath}</color>");
            return;
        }

        var testQuestions = JsonUtility.FromJson<TestBankWrapper>(json);

        if (testQuestions == null || testQuestions.test_bank == null)
        {
            Debug.LogError("[ClassroomManager] Test bank JSON parsed to null.");
            return;
        }

        var payload = new TestBankPayload
        {
            test_bank = testQuestions.test_bank,
            max_questions_per_minute = maxQuestionsPerMinute
        };

        StartCoroutine(SendQuestioningInit(url, payload));
    }

    // Posts the test bank payload to the backend questioning initialization endpoint
    IEnumerator SendQuestioningInit(string url, TestBankPayload payload)
    {
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
                Debug.Log("<color=green>Questioning system initialized</color>");
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PRE-RECORDED LECTURE PROCESSING
    // ─────────────────────────────────────────────────────────────────────────

    // Converts a pre-recorded AudioClip to 16 kHz mono WAV and uploads it
    // to the Whisper transcription service, then sends the result to all students.
    IEnumerator ProcessPreRecordedLecture()
    {
        if (preRecordedLecture == null)
        {
            yield break;
        }

        Debug.Log($"<color=cyan>Processing pre-recorded lecture...</color>");
        Debug.Log($"  Original: {preRecordedLecture.frequency}Hz, {preRecordedLecture.channels}ch, {preRecordedLecture.length:F1}s");

        // Extract raw float samples from the AudioClip
        float[] samples = new float[preRecordedLecture.samples * preRecordedLecture.channels];
        preRecordedLecture.GetData(samples, 0);

        // Whisper works best at 16 kHz; downsample from Unity's 44.1 kHz
        int targetSampleRate = 16000;
        float[] downsampledSamples = DownsampleAudio(samples, preRecordedLecture.frequency, targetSampleRate, preRecordedLecture.channels);
        float[] monoSamples = ConvertToMono(downsampledSamples, preRecordedLecture.channels);
        byte[] wavData = ConvertAudioClipToWav(monoSamples, 1, targetSampleRate);

        Debug.Log($"  Final size: {wavData.Length / (1024f * 1024f):F2} MB");

        WWWForm form = new WWWForm();
        form.AddBinaryData("file", wavData, "prerecorded.wav");
        form.AddField("is_snippet", "False");

        using (UnityWebRequest www = UnityWebRequest.Post("http://127.0.0.1:5005/transcribe", form))
        {
            www.timeout = 300;   // allow up to 5 minutes for a long lecture
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                var response = JsonUtility.FromJson<VoiceResponse>(www.downloadHandler.text);
                loadedLectureText = response.cleaned;
                Debug.Log($"<color=green>✓ Transcription complete!</color>");
                SendLectureToAll();
            }
            else
            {
                Debug.LogError($"Transcription failed: {www.error}");
            }
        }
    }

    // Resamples audio from originalRate to targetRate using linear interpolation.
    // Returns the original array unchanged if both rates are already the same.
    float[] DownsampleAudio(float[] samples, int originalRate, int targetRate, int channels)
    {
        if (originalRate == targetRate)
        {
            return samples;
        }

        // The ratio tells us how many original samples correspond to one target sample
        float ratio = (float)originalRate / targetRate;
        int newLength = Mathf.RoundToInt(samples.Length / ratio);
        float[] down = new float[newLength];

        for (int i = 0; i < newLength; i++)
        {
            // Map target index back to a fractional position in the original array
            float srcIndex = i * ratio;
            int index = Mathf.FloorToInt(srcIndex);

            if (index < samples.Length)
            {
                float frac = srcIndex - index;

                if (index + 1 < samples.Length)
                {
                    // Linear interpolation between adjacent original samples
                    down[i] = Mathf.Lerp(samples[index], samples[index + 1], frac);
                }
                else
                {
                    down[i] = samples[index];
                }
            }
        }

        return down;
    }

    // Mixes a multi-channel audio array down to mono by averaging all channels per frame.
    // Returns the input unchanged if it's already mono.
    float[] ConvertToMono(float[] samples, int channels)
    {
        if (channels == 1)
        {
            return samples;
        }

        int monoLength = samples.Length / channels;
        float[] mono = new float[monoLength];

        for (int i = 0; i < monoLength; i++)
        {
            float sum = 0;
            for (int c = 0; c < channels; c++)
            {
                sum += samples[i * channels + c];
            }
            mono[i] = sum / channels;
        }

        return mono;
    }

    // Writes a minimal WAV file (RIFF header + PCM data) from a float sample array.
    // Used for pre-recorded lectures before uploading to Whisper.
    byte[] ConvertAudioClipToWav(float[] samples, int channels, int frequency)
    {
        MemoryStream stream = new MemoryStream();
        BinaryWriter writer = new BinaryWriter(stream);

        // RIFF chunk descriptor
        writer.Write("RIFF".ToCharArray());
        writer.Write(36 + samples.Length * 2);   // total file size minus 8 bytes
        writer.Write("WAVE".ToCharArray());

        // fmt sub-chunk
        writer.Write("fmt ".ToCharArray());
        writer.Write(16);                         // fmt chunk size (always 16 for PCM)
        writer.Write((short)1);                   // PCM format identifier
        writer.Write((short)channels);
        writer.Write(frequency);
        writer.Write(frequency * channels * 2);   // byte rate
        writer.Write((short)(channels * 2));       // block align
        writer.Write((short)16);                   // bits per sample

        // data sub-chunk
        writer.Write("data".ToCharArray());
        writer.Write(samples.Length * 2);          // data size in bytes

        // Convert floats to signed 16-bit PCM integers
        foreach (var sample in samples)
        {
            writer.Write((short)(sample * 32767));
        }

        return stream.ToArray();
    }

    // Inner class matching the JSON shape returned by Agent_Voice.py
    [System.Serializable]
    private class VoiceResponse
    {
        public string original;   // raw Whisper output before filler removal
        public string cleaned;    // cleaned version fed to students
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SESSION DATA EXPORT
    // ─────────────────────────────────────────────────────────────────────────

    // Builds the full session data object and writes it as both JSON and a
    // human-readable text report to the given file path.
    public void ExportSessionData(string filepath)
    {
        // Build per-tier statistics from the collected score lists
        Dictionary<string, TierStatistics> tierStats = new Dictionary<string, TierStatistics>();

        foreach (var entry in tierScores)
        {
            if (entry.Value.Count == 0)
            {
                continue;
            }

            List<int> scores = new List<int>(entry.Value);
            scores.Sort();

            float mean = (float)scores.Average();
            float median = scores[scores.Count / 2];
            float variance = (float)scores.Select(s => Mathf.Pow(s - mean, 2)).Average();
            float stdDev = Mathf.Sqrt(variance);

            tierStats[entry.Key] = new TierStatistics
            {
                tier = entry.Key,
                studentCount = scores.Count,
                scores = scores,
                mean = mean,
                median = median,
                stdDev = stdDev,
                min = scores.Min(),
                max = scores.Max()
            };
        }

        // Build the per-student result list for the export
        List<StudentResult> studentResults = new List<StudentResult>();
        foreach (var student in students)
        {
            int score = 0;
            if (tierScores.ContainsKey(student.tier))
            {
                var list = tierScores[student.tier];
                if (list.Count > 0)
                {
                    score = list[0];
                }
            }

            studentResults.Add(new StudentResult
            {
                studentId = student.studentID,
                studentName = student.studentName,
                tier = student.tier,
                score = score
            });
        }

        var sessionData = new SessionData
        {
            sessionDate = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
            totalStudents = students.Count,
            tierStatistics = tierStats,
            studentResults = studentResults,
            questionTimeline = questionTimeline,
            lectureSnippet = currentLectureSnippet,
            examPath = GetExamFilePath(),
            testBankPath = GetTestBankFilePath()
        };

        string json = JsonUtility.ToJson(sessionData, true);
        System.IO.File.WriteAllText(filepath, json);

        Debug.Log($"<color=green>╔══════════════════════════════════════════════════╗</color>");
        Debug.Log($"<color=green>║ SESSION RESULTS EXPORTED</color>");
        Debug.Log($"<color=green>╚══════════════════════════════════════════════════╝</color>");
        Debug.Log($"<color=cyan>File: {filepath}</color>");
        Debug.Log($"<color=cyan>Students: {students.Count} | Questions Asked: {questionTimeline.Count}</color>");

        // Also write a plain-text version next to the JSON for quick human reading
        ExportTextReport(filepath.Replace(".json", ".txt"), sessionData, tierStats);
    }

    // Writes a formatted plain-text report with tier tables, question log, and paths
    void ExportTextReport(string filepath, SessionData data, Dictionary<string, TierStatistics> tierStats)
    {
        System.Text.StringBuilder report = new System.Text.StringBuilder();

        report.AppendLine("═══════════════════════════════════════════════════════════");
        report.AppendLine("              VR CLASSROOM SESSION REPORT");
        report.AppendLine("═══════════════════════════════════════════════════════════");
        report.AppendLine();
        report.AppendLine($"Session Date: {data.sessionDate}");
        report.AppendLine($"Total Students: {data.totalStudents}");
        report.AppendLine();
        report.AppendLine("───────────────────────────────────────────────────────────");
        report.AppendLine("TIER PERFORMANCE SUMMARY");
        report.AppendLine("───────────────────────────────────────────────────────────");
        report.AppendLine();
        report.AppendLine("Tier      Students  Mean   Median  StdDev  Min  Max");
        report.AppendLine("─────────────────────────────────────────────────────────");

        float overallMean = 0f;
        int totalCount = 0;

        foreach (var stat in tierStats.OrderBy(x => x.Key))
        {
            var s = stat.Value;
            report.AppendLine($"{s.tier,-10}{s.studentCount,-10}{s.mean,-7:F1}{s.median,-8:F1}{s.stdDev,-8:F2}{s.min,-5}{s.max,-5}");
            overallMean += s.mean * s.studentCount;
            totalCount += s.studentCount;
        }

        if (totalCount > 0)
        {
            overallMean /= totalCount;
            report.AppendLine("─────────────────────────────────────────────────────────");
            report.AppendLine($"Overall Mean Score: {overallMean:F2}");
        }

        report.AppendLine();
        report.AppendLine("───────────────────────────────────────────────────────────");
        report.AppendLine("QUESTIONS ASKED DURING SESSION");
        report.AppendLine("───────────────────────────────────────────────────────────");
        report.AppendLine();

        if (data.questionTimeline.Count == 0)
        {
            report.AppendLine("No questions were asked during this session.");
        }
        else
        {
            foreach (var q in data.questionTimeline)
            {
                float minutes = q.timestamp / 60f;
                report.AppendLine($"[{minutes:F1}m] {q.studentName} ({q.questionType})");
                report.AppendLine($"  Q: {q.question}");
                report.AppendLine();
            }
        }

        report.AppendLine("───────────────────────────────────────────────────────────");
        report.AppendLine("CURRENT LECTURE CONTEXT");
        report.AppendLine("───────────────────────────────────────────────────────────");
        report.AppendLine();

        if (string.IsNullOrEmpty(data.lectureSnippet))
        {
            report.AppendLine("No lecture snippet recorded.");
        }
        else
        {
            report.AppendLine(data.lectureSnippet);
        }

        report.AppendLine();
        report.AppendLine("───────────────────────────────────────────────────────────");
        report.AppendLine("FILE PATHS");
        report.AppendLine("───────────────────────────────────────────────────────────");
        report.AppendLine($"Exam:      {data.examPath}");
        report.AppendLine($"Test Bank: {data.testBankPath}");
        report.AppendLine();
        report.AppendLine("═══════════════════════════════════════════════════════════");
        report.AppendLine("              END OF REPORT");
        report.AppendLine("═══════════════════════════════════════════════════════════");

        System.IO.File.WriteAllText(filepath, report.ToString());
        Debug.Log($"<color=cyan>Text report: {filepath}</color>");
    }

    // Returns to the main menu scene via the scene transition manager
    public void GoToMainMenu()
    {
        SceneTransitionManager.singleton.GoToSceneAsync(0);
    }
}

// ==================== DATA STRUCTURES ====================
// All these are declared outside ClassroomManager so JsonUtility and other
// scripts can reference them without needing a ClassroomManager instance.

[System.Serializable]
public class StudentQuestionResponse
{
    public bool has_question;
    public string question;
    public string question_type;
    public string student_name;
    public float confidence;
}

[System.Serializable]
public class TeacherAnswerResponse
{
    public string status;
    public string message;
    public string question;
    public string answer;
    public float confidence;
    public float processing_time;
}

[System.Serializable]
public class QuestionEvent
{
    public float timestamp;
    public string studentName;
    public string question;
    public string questionType;
}

[System.Serializable]
public class TestBankPayload
{
    public List<TestQuestion> test_bank;
    public int max_questions_per_minute;
}

[System.Serializable]
public class TestQuestion
{
    public string text;
    public List<string> options;
}

[System.Serializable]
public class TestBankWrapper
{
    public List<TestQuestion> test_bank;
}

[System.Serializable]
public class SessionData
{
    public string sessionDate;
    public int totalStudents;
    public Dictionary<string, TierStatistics> tierStatistics;
    public List<StudentResult> studentResults;
    public List<QuestionEvent> questionTimeline;
    public string lectureSnippet;
    public string examPath;
    public string testBankPath;
}

[System.Serializable]
public class TierStatistics
{
    public string tier;
    public int studentCount;
    public List<int> scores;
    public float mean;
    public float median;
    public float stdDev;
    public int min;
    public int max;
}

[System.Serializable]
public class StudentResult
{
    public int studentId;
    public string studentName;
    public string tier;
    public int score;
}

[System.Serializable]
public class ExamSheet
{
    public List<ExamQuestion> questions;
}

[System.Serializable]
public class ExamQuestion
{
    public string text;
    public List<string> options;
    public int correctIndex;   // 0-based index of the correct option
}