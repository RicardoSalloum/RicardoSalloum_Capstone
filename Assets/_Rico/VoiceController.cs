using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.IO;

// VoiceController handles all microphone recording and audio uploading.
// It records live audio from the instructor's mic, sends 30-second snippets
// to the Whisper transcription service during the lecture, and uploads the
// full recording when the lecture ends.
public class VoiceController : MonoBehaviour
{
    [Header("Configuration")]
    public string voiceAgentUrl = "http://127.0.0.1:5005/transcribe";
    public ClassroomManager classroomManager;

    // How often (in seconds) to send a chunk of audio to Whisper during the lecture
    public float snippetInterval = 30f;

    private AudioClip lectureRecording;
    private bool isLectureActive = false;
    private string micName;
    private float lastSnippetTime = 0f;

    // True while any WAV upload (snippet or full) is still waiting on the server.
    // ClassroomManager polls this before starting the exam so it doesn't race the
    // final Whisper response.
    private int _activeUploads = 0;
    public bool IsUploadInFlight => _activeUploads > 0;

    // We record at 44100 Hz because that's Unity's standard mic sample rate
    private const int SampleRate = 44100;

    void Start()
    {
        // Grab the first available microphone; warn if none is detected
        if (Microphone.devices.Length > 0)
        {
            micName = Microphone.devices[0];
        }
        else
        {
            Debug.LogError("No Microphone detected! Ensure a mic is plugged in.");
        }
    }

    void Update()
    {
        // While a lecture is active, check whether it's time to send the next snippet
        if (isLectureActive)
        {
            if (Time.time - lastSnippetTime >= snippetInterval)
            {
                SendSnippet();
                lastSnippetTime = Time.time;
            }
        }
    }

    // Read-only property so ClassroomManager can check the lecture state
    public bool IsLectureActive
    {
        get { return isLectureActive; }
    }

    // Toggles the lecture on or off and starts/stops the questioning system accordingly
    public void ToggleLecture()
    {
        if (isLectureActive)
        {
            StopFullLecture();

            // Stop questioning and TTS through the manager so the _lectureActive
            // flag is cleared before any in-flight coroutines check it
            if (classroomManager != null)
            {
                classroomManager.StopQuestioningSystem();
            }
        }
        else
        {
            StartFullLecture();
            classroomManager.StartQuestioningSystem();
        }
    }

    // Begins microphone recording and resets the snippet timer.
    // 1800 seconds (30 minutes) avoids Unity's internal one-hour AudioClip size limit.
    public void StartFullLecture()
    {
        if (isLectureActive)
        {
            return;
        }

        isLectureActive = true;
        lastSnippetTime = Time.time;
        lectureRecording = Microphone.Start(micName, false, 1800, SampleRate);

        Debug.Log("<color=green><b>LECTURE STARTED:</b> Recording and periodic snippets active.</color>");
    }

    // Stops the microphone, slices out the recorded samples, and uploads
    // the full recording to Whisper for final transcription.
    public void StopFullLecture()
    {
        if (!isLectureActive)
        {
            return;
        }

        // GetPosition returns the number of samples written so far — used to trim silence
        int finalPos = Microphone.GetPosition(micName);
        Microphone.End(micName);
        isLectureActive = false;

        // Only upload if any audio was actually recorded
        if (finalPos > 0)
        {
            float[] finalSamples = new float[finalPos];
            lectureRecording.GetData(finalSamples, 0);
            Normalize(finalSamples);

            byte[] wavData = ConvertToWavFromSamples(finalSamples, 1);
            StartCoroutine(UploadVoice(wavData, false));
        }

        Debug.Log("<color=red><b>LECTURE STOPPED:</b> Finalizing full recording.</color>");
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    // Extracts the audio from the last 'snippetInterval' seconds and uploads it.
    // Runs every snippetInterval seconds while the lecture is active.
    void SendSnippet()
    {
        int currentPos = Microphone.GetPosition(micName);
        int samplesNeeded = (int)(snippetInterval * SampleRate);

        // Don't send if less than one full snippet interval has been recorded yet
        if (currentPos < samplesNeeded)
        {
            return;
        }

        float[] snippetSamples = new float[samplesNeeded];

        // Calculate the start offset so we read the most recent 'samplesNeeded' samples.
        // Clamped to 0 to avoid negative offsets when currentPos is barely >= samplesNeeded.
        int startSample = Mathf.Max(0, currentPos - samplesNeeded);
        lectureRecording.GetData(snippetSamples, startSample);

        Normalize(snippetSamples);

        byte[] wavData = ConvertToWavFromSamples(snippetSamples, 1);
        StartCoroutine(UploadVoice(wavData, true));
    }

    // Scales all samples so the loudest one is exactly 1.0, preventing clipping.
    void Normalize(float[] samples)
    {
        float max = 0;

        // Find the peak absolute value across all samples
        foreach (var s in samples)
        {
            if (Mathf.Abs(s) > max)
            {
                max = Mathf.Abs(s);
            }
        }

        // Avoid division by zero; only normalize if the audio is non-silent
        if (max > 0)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] /= max;
            }
        }
    }

    // Posts the WAV bytes to the transcription service.
    // On success, appends the cleaned transcript to the classroom's lecture text
    // so the student agents always have up-to-date context.
    IEnumerator UploadVoice(byte[] data, bool isSnippet)
    {
        _activeUploads++;

        WWWForm form = new WWWForm();
        form.AddBinaryData("file", data, isSnippet ? "snippet.wav" : "full_lecture.wav");
        form.AddField("is_snippet", isSnippet.ToString());

        using (UnityWebRequest www = UnityWebRequest.Post(voiceAgentUrl, form))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                var response = JsonUtility.FromJson<VoiceResponse>(www.downloadHandler.text);

                if (isSnippet)
                {
                    // Discard snippet results that arrive after the lecture was stopped.
                    // The coroutine was already in-flight when ToggleLecture() ran,
                    // so we guard here rather than trying to cancel the web request.
                    if (!isLectureActive)
                    {
                        Debug.Log("<color=grey>Snippet discarded — lecture already stopped.</color>");
                        _activeUploads--;
                        yield break;
                    }

                    Debug.Log("<color=yellow>Snippet:</color> " + response.cleaned);

                    // Append the new snippet text to the running lecture transcript
                    classroomManager.loadedLectureText += " " + response.cleaned;

                    // Also update the QuestioningManager's snippet so it always
                    // generates questions based on the freshest lecture content
                    classroomManager.UpdateCurrentSnippet(classroomManager.loadedLectureText);
                }
                else
                {
                    Debug.Log("<color=cyan>Full Lecture Received!</color>");

                    // Full transcription replaces whatever partial text we had
                    classroomManager.loadedLectureText = response.cleaned;
                    classroomManager.UpdateCurrentSnippet(classroomManager.loadedLectureText);
                }
            }
        }

        _activeUploads--;
    }

    // Builds a valid WAV file from raw float sample data.
    // Writes the RIFF/WAVE header manually, then converts floats to 16-bit PCM.
    byte[] ConvertToWavFromSamples(float[] samples, int channels)
    {
        MemoryStream stream = new MemoryStream();
        BinaryWriter writer = new BinaryWriter(stream);

        // RIFF header
        writer.Write("RIFF".ToCharArray());
        writer.Write(36 + samples.Length * 2);   // file size minus the 8-byte RIFF header

        // WAVE + fmt chunk
        writer.Write("WAVE".ToCharArray());
        writer.Write("fmt ".ToCharArray());
        writer.Write(16);                          // fmt chunk size (always 16 for PCM)
        writer.Write((short)1);                    // PCM format
        writer.Write((short)channels);
        writer.Write(SampleRate);
        writer.Write(SampleRate * channels * 2);   // byte rate
        writer.Write((short)(channels * 2));        // block align
        writer.Write((short)16);                    // bits per sample

        // data chunk
        writer.Write("data".ToCharArray());
        writer.Write(samples.Length * 2);           // data size in bytes

        // Convert each float (-1.0 to 1.0) to a signed 16-bit integer
        foreach (var sample in samples)
        {
            writer.Write((short)(sample * 32767));
        }

        return stream.ToArray();
    }

    // Simple inner class matching the JSON shape returned by Agent_Voice.py
    [System.Serializable]
    private class VoiceResponse
    {
        public string original;   // raw Whisper output
        public string cleaned;    // filler-word-stripped version
    }
}