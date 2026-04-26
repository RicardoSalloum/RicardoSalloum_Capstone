using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;
using System.IO;

public class SimpleSlideshow : MonoBehaviour
{
    [Header("UI References")]
    public RawImage slideDisplay;
    public Button nextButton;
    public Button previousButton;
    public TextMeshProUGUI slideCounter;

    [Header("Slide Settings")]
    [Tooltip("Fallback folder path used only if no PlayerPref has been saved yet.")]
    public string folderPath = "C:/Slides";

    // PlayerPrefs key — shared with PathSettingsMenu so both scripts agree on the key name.
    public const string PREF_SLIDES_FOLDER = "SlidesFolderPath";

    // Private
    private List<Texture2D> slides = new List<Texture2D>();
    private int currentSlide = 0;

    void Start()
    {
        // Load the saved path; fall back to the inspector default if nothing is saved yet.
        folderPath = PlayerPrefs.GetString(PREF_SLIDES_FOLDER, folderPath);

        if (nextButton != null)
            nextButton.onClick.AddListener(NextSlide);

        if (previousButton != null)
            previousButton.onClick.AddListener(PreviousSlide);

        UpdateButtons();
        LoadSlides();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.Space))
            NextSlide();

        if (Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.Backspace))
            PreviousSlide();

        if (Input.GetKeyDown(KeyCode.Home))
            GoToSlide(0);

        if (Input.GetKeyDown(KeyCode.End))
            GoToSlide(slides.Count - 1);
    }

    // ==================== PUBLIC METHODS ====================

    /// <summary>
    /// Hot-swap the slides folder at runtime. Saves the path and reloads immediately.
    /// Called by PathSettingsMenu when the user confirms a new path.
    /// </summary>
    public void SetFolderPath(string newPath)
    {
        folderPath = newPath;
        PlayerPrefs.SetString(PREF_SLIDES_FOLDER, newPath);
        PlayerPrefs.Save();
        LoadSlides();
    }

    public void LoadSlides()
    {
        if (!Directory.Exists(folderPath))
        {
            Debug.LogError($"Folder not found: {folderPath}");
            return;
        }

        Debug.Log($"Loading slides from: {folderPath}");

        ClearSlides();

        string[] allFiles = Directory.GetFiles(folderPath);
        Debug.Log($"  Total files in folder: {allFiles.Length}");

        List<string> imageFiles = new List<string>();
        foreach (var file in allFiles)
        {
            string ext = Path.GetExtension(file).ToLower();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
                imageFiles.Add(file);
        }

        Debug.Log($"  Image files found: {imageFiles.Count}");

        if (imageFiles.Count > 0)
            Debug.Log($"  First file: {Path.GetFileName(imageFiles[0])}");
        else if (allFiles.Length > 0)
        {
            Debug.LogWarning($"  No image files found, but folder has {allFiles.Length} files");
            Debug.LogWarning($"  First file in folder: {Path.GetFileName(allFiles[0])}");
        }

        imageFiles.Sort();

        if (imageFiles.Count == 0)
        {
            Debug.LogError($"No PNG/JPG images found in {folderPath}");
            return;
        }

        foreach (var filePath in imageFiles)
        {
            byte[] fileData = File.ReadAllBytes(filePath);
            Texture2D texture = new Texture2D(2, 2);

            if (texture.LoadImage(fileData))
            {
                slides.Add(texture);
                Debug.Log($"  Loaded {Path.GetFileName(filePath)}");
            }
            else
            {
                Debug.LogWarning($"  Failed to load {Path.GetFileName(filePath)}");
                Destroy(texture);
            }
        }

        currentSlide = 0;
        ShowCurrentSlide();
        UpdateButtons();

        Debug.Log($"Loaded {slides.Count} slides");
    }

    public void NextSlide()
    {
        if (currentSlide < slides.Count - 1)
        {
            currentSlide++;
            ShowCurrentSlide();
            UpdateButtons();
        }
    }

    public void PreviousSlide()
    {
        if (currentSlide > 0)
        {
            currentSlide--;
            ShowCurrentSlide();
            UpdateButtons();
        }
    }

    public void GoToSlide(int slideIndex)
    {
        if (slideIndex >= 0 && slideIndex < slides.Count)
        {
            currentSlide = slideIndex;
            ShowCurrentSlide();
            UpdateButtons();
        }
    }

    void ShowCurrentSlide()
    {
        if (slideDisplay != null && slides.Count > 0 && currentSlide < slides.Count)
            slideDisplay.texture = slides[currentSlide];

        UpdateSlideCounter();
    }

    void UpdateButtons()
    {
        if (previousButton != null)
            previousButton.interactable = currentSlide > 0;

        if (nextButton != null)
            nextButton.interactable = currentSlide < slides.Count - 1;

        UpdateSlideCounter();
    }

    void UpdateSlideCounter()
    {
        if (slideCounter != null)
        {
            slideCounter.text = slides.Count > 0
                ? $"{currentSlide + 1} / {slides.Count}"
                : "No slides loaded";
        }
    }

    void ClearSlides()
    {
        foreach (var slide in slides)
        {
            if (slide != null)
                Destroy(slide);
        }
        slides.Clear();
    }

    void OnDestroy()
    {
        ClearSlides();
    }

    public int GetCurrentSlideNumber() => currentSlide + 1;
    public int GetTotalSlides() => slides.Count;
    public bool HasSlides() => slides.Count > 0;
}