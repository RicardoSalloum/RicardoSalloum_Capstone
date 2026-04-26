using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.IO;

/// <summary>
/// Main Menu settings panel for VR - InputField version.
/// Users type paths directly into TMP_InputFields.
///
/// SETUP:
///   1. Attach to your settings panel GameObject.
///   2. Wire all Inspector references below.
///
/// PANEL HIERARCHY SUGGESTION:
///   SettingsPanel
///   ├─ SlidesRow
///   │  ├─ SlidesLabel          (TMP — "Slides Folder")
///   │  └─ SlidesInputField     ← slidesFolderInput
///   ├─ ExamRow
///   │  ├─ ExamLabel            (TMP — "Exam File")
///   │  └─ ExamInputField       ← examFileInput
///   ├─ TestBankRow
///   │  ├─ TestBankLabel        (TMP — "Test Bank File")
///   │  └─ TestBankInputField   ← testBankFileInput
///   ├─ SaveButton              ← saveButton
///   ├─ StatusLabel             ← statusLabel
///   └─ ClearButton             ← clearAllButton (optional)
/// </summary>
public class PathSettingsMenu : MonoBehaviour
{
    // ── PlayerPrefs keys ──────────────────────────────────────────────────────
    public const string PREF_SLIDES_FOLDER = "SlidesFolderPath";
    public const string PREF_EXAM_FILE = "ExamDataFilePath";
    public const string PREF_TESTBANK_FILE = "TestBankFilePath";

    [Header("Input Fields")]
    [Tooltip("InputField for slides folder path")]
    public TMP_InputField slidesFolderInput;

    [Tooltip("InputField for exam data file path")]
    public TMP_InputField examFileInput;

    [Tooltip("InputField for test bank file path")]
    public TMP_InputField testBankFileInput;

    [Header("Actions")]
    public Button saveButton;
    public Button clearAllButton;

    [Header("Feedback")]
    public TextMeshProUGUI statusLabel;

    [Header("Defaults (fallback when no pref saved yet)")]
    public string defaultSlidesFolder = "C:/Slides";
    public string defaultExamFile = "C:/ClassroomData/exam.json";
    public string defaultTestBankFile = "C:/ClassroomData/testbank.json";

    // ─────────────────────────────────────────────────────────────────────────

    void Start()
    {
        LoadSavedPaths();

        if (saveButton != null)
            saveButton.onClick.AddListener(SaveAllPaths);

        if (clearAllButton != null)
            clearAllButton.onClick.AddListener(ClearAll);

        // Add validation listeners to input fields
        if (slidesFolderInput != null)
            slidesFolderInput.onEndEdit.AddListener(_ => ValidateSlidesFolderPath());

        if (examFileInput != null)
            examFileInput.onEndEdit.AddListener(_ => ValidateExamFilePath());

        if (testBankFileInput != null)
            testBankFileInput.onEndEdit.AddListener(_ => ValidateTestBankFilePath());

        SetStatus("");
    }

    // ── Load saved paths into input fields ───────────────────────────────────

    void LoadSavedPaths()
    {
        if (slidesFolderInput != null)
        {
            string savedPath = PlayerPrefs.GetString(PREF_SLIDES_FOLDER, defaultSlidesFolder);
            slidesFolderInput.text = savedPath;
            ValidateSlidesFolderPath();
        }

        if (examFileInput != null)
        {
            string savedPath = PlayerPrefs.GetString(PREF_EXAM_FILE, defaultExamFile);
            examFileInput.text = savedPath;
            ValidateExamFilePath();
        }

        if (testBankFileInput != null)
        {
            string savedPath = PlayerPrefs.GetString(PREF_TESTBANK_FILE, defaultTestBankFile);
            testBankFileInput.text = savedPath;
            ValidateTestBankFilePath();
        }

        Debug.Log("<color=cyan>[PathSettingsMenu] Loaded saved paths</color>");
    }

    // ── Save all paths to PlayerPrefs ─────────────────────────────────────────

    void SaveAllPaths()
    {
        bool allValid = true;
        int savedCount = 0;

        // Save slides folder
        if (slidesFolderInput != null && !string.IsNullOrEmpty(slidesFolderInput.text))
        {
            string path = slidesFolderInput.text.Trim();
            if (Directory.Exists(path))
            {
                PlayerPrefs.SetString(PREF_SLIDES_FOLDER, path);
                savedCount++;
            }
            else
            {
                allValid = false;
                Debug.LogWarning($"[PathSettingsMenu] Slides folder does not exist: {path}");
            }
        }

        // Save exam file
        if (examFileInput != null && !string.IsNullOrEmpty(examFileInput.text))
        {
            string path = examFileInput.text.Trim();
            if (File.Exists(path))
            {
                PlayerPrefs.SetString(PREF_EXAM_FILE, path);
                savedCount++;
            }
            else
            {
                allValid = false;
                Debug.LogWarning($"[PathSettingsMenu] Exam file does not exist: {path}");
            }
        }

        // Save test bank file
        if (testBankFileInput != null && !string.IsNullOrEmpty(testBankFileInput.text))
        {
            string path = testBankFileInput.text.Trim();
            if (File.Exists(path))
            {
                PlayerPrefs.SetString(PREF_TESTBANK_FILE, path);
                savedCount++;
            }
            else
            {
                allValid = false;
                Debug.LogWarning($"[PathSettingsMenu] Test bank file does not exist: {path}");
            }
        }

        PlayerPrefs.Save();

        if (allValid && savedCount > 0)
        {
            SetStatus($"✓ Saved {savedCount} path(s) successfully!", false);
            Debug.Log($"<color=green>[PathSettingsMenu] Saved {savedCount} paths</color>");
        }
        else if (savedCount > 0)
        {
            SetStatus($"⚠ Saved {savedCount} path(s), but some paths are invalid", true);
        }
        else
        {
            SetStatus("⚠ No valid paths to save", true);
        }

        // Refresh validation colors
        ValidateSlidesFolderPath();
        ValidateExamFilePath();
        ValidateTestBankFilePath();
    }

    // ── Validation methods ────────────────────────────────────────────────────

    void ValidateSlidesFolderPath()
    {
        if (slidesFolderInput == null) return;

        string path = slidesFolderInput.text.Trim();
        bool valid = !string.IsNullOrEmpty(path) && Directory.Exists(path);

        // Change input field text color based on validity
        if (slidesFolderInput.textComponent != null)
        {
            slidesFolderInput.textComponent.color = string.IsNullOrEmpty(path)
                ? Color.gray
                : (valid ? Color.green : Color.red);
        }
    }

    void ValidateExamFilePath()
    {
        if (examFileInput == null) return;

        string path = examFileInput.text.Trim();
        bool valid = !string.IsNullOrEmpty(path) && File.Exists(path);

        if (examFileInput.textComponent != null)
        {
            examFileInput.textComponent.color = string.IsNullOrEmpty(path)
                ? Color.gray
                : (valid ? Color.green : Color.red);
        }
    }

    void ValidateTestBankFilePath()
    {
        if (testBankFileInput == null) return;

        string path = testBankFileInput.text.Trim();
        bool valid = !string.IsNullOrEmpty(path) && File.Exists(path);

        if (testBankFileInput.textComponent != null)
        {
            testBankFileInput.textComponent.color = string.IsNullOrEmpty(path)
                ? Color.gray
                : (valid ? Color.green : Color.red);
        }
    }

    // ── Helper methods ────────────────────────────────────────────────────────

    void SetStatus(string message, bool error = false)
    {
        if (statusLabel == null) return;
        statusLabel.text = message;
        statusLabel.color = error ? Color.red : Color.green;
    }

    public void ClearAll()
    {
        PlayerPrefs.DeleteKey(PREF_SLIDES_FOLDER);
        PlayerPrefs.DeleteKey(PREF_EXAM_FILE);
        PlayerPrefs.DeleteKey(PREF_TESTBANK_FILE);
        PlayerPrefs.Save();

        // Reset input fields to defaults
        if (slidesFolderInput != null)
        {
            slidesFolderInput.text = defaultSlidesFolder;
            ValidateSlidesFolderPath();
        }

        if (examFileInput != null)
        {
            examFileInput.text = defaultExamFile;
            ValidateExamFilePath();
        }

        if (testBankFileInput != null)
        {
            testBankFileInput.text = defaultTestBankFile;
            ValidateTestBankFilePath();
        }

        SetStatus("All paths reset to defaults.");
        Debug.Log("<color=yellow>[PathSettingsMenu] All paths cleared and reset to defaults</color>");
    }

    // ── Public methods for other scripts ──────────────────────────────────────

    /// <summary>Get the currently saved slides folder path</summary>
    public string GetSlidesFolderPath()
    {
        return PlayerPrefs.GetString(PREF_SLIDES_FOLDER, defaultSlidesFolder);
    }

    /// <summary>Get the currently saved exam file path</summary>
    public string GetExamFilePath()
    {
        return PlayerPrefs.GetString(PREF_EXAM_FILE, defaultExamFile);
    }

    /// <summary>Get the currently saved test bank file path</summary>
    public string GetTestBankFilePath()
    {
        return PlayerPrefs.GetString(PREF_TESTBANK_FILE, defaultTestBankFile);
    }
}