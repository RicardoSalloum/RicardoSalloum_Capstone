using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(ClassroomManager))]
public class ClassroomManagerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        ClassroomManager classroom = (ClassroomManager)target;

        GUILayout.Space(15);
        GUILayout.Label("Simulation Controls", EditorStyles.boldLabel);

        // Green button for the Lecture
        GUI.backgroundColor = Color.green;
        if (GUILayout.Button("1. Send Lecture to All", GUILayout.Height(30)))
        {
            classroom.SendLectureToAll();
        }

        // Cyan button for starting questioning
        GUI.backgroundColor = Color.cyan;
        if (GUILayout.Button("2. Start Questioning System", GUILayout.Height(30)))
        {
            classroom.StartQuestioningSystem();
        }

        // Standard button for the single question
        GUI.backgroundColor = Color.white;
        if (GUILayout.Button("3. Send Single Question (Live QA)", GUILayout.Height(30)))
        {
            classroom.AskAllStudents();
        }

        GUILayout.Space(10);

        // Blue button for the JSON test
        GUI.backgroundColor = new Color(0.5f, 0.8f, 1f);
        if (GUILayout.Button("4. START TESTING PHASE (JSON Exam)", GUILayout.Height(40)))
        {
            classroom.TriggerTestingPhase();
        }

        GUI.backgroundColor = Color.white;
        GUILayout.Space(15);
        GUILayout.Label("Debug Tools", EditorStyles.boldLabel);

        // Yellow button for debugging exam data
        GUI.backgroundColor = Color.yellow;
        if (GUILayout.Button("Debug: Show Exam Questions", GUILayout.Height(25)))
        {
            classroom.DebugExamData();
        }

        GUI.backgroundColor = Color.white;

        // --- Exam file path status ---
        string examPath = PlayerPrefs.GetString(PathSettingsMenu.PREF_EXAM_FILE, classroom.examDataFallbackPath);
        bool examExists = !string.IsNullOrEmpty(examPath) && System.IO.File.Exists(examPath);

        if (!examExists)
        {
            EditorGUILayout.HelpBox(
                "⚠️ Exam file not found!\n\n" +
                $"Current path: {(string.IsNullOrEmpty(examPath) ? "(empty)" : examPath)}\n\n" +
                "Set the correct path in the Main Menu via PathSettingsMenu,\n" +
                "or update 'Exam Data Fallback Path' in the inspector above.",
                MessageType.Warning
            );
        }
        else
        {
            EditorGUILayout.HelpBox(
                $"✓ Exam file found:\n{examPath}",
                MessageType.Info
            );
        }

        // --- Test bank file path status ---
        string testBankPath = PlayerPrefs.GetString(PathSettingsMenu.PREF_TESTBANK_FILE, classroom.testBankFallbackPath);
        bool testBankExists = !string.IsNullOrEmpty(testBankPath) && System.IO.File.Exists(testBankPath);

        if (!testBankExists)
        {
            EditorGUILayout.HelpBox(
                "⚠️ Test bank file not found!\n\n" +
                $"Current path: {(string.IsNullOrEmpty(testBankPath) ? "(empty)" : testBankPath)}\n\n" +
                "Set the correct path in the Main Menu via PathSettingsMenu,\n" +
                "or update 'Test Bank Fallback Path' in the inspector above.",
                MessageType.Warning
            );
        }
        else
        {
            EditorGUILayout.HelpBox(
                $"✓ Test bank file found:\n{testBankPath}",
                MessageType.Info
            );
        }

        // --- QuestioningManager status ---
        if (classroom.questioningManager == null)
        {
            EditorGUILayout.HelpBox(
                "⚠️ QuestioningManager not assigned!\n\n" +
                "Add QuestioningManager component to the scene and assign it above.",
                MessageType.Warning
            );
        }
    }
}