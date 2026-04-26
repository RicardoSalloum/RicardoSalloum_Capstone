using System.Collections.Generic;
using System.Linq;
using UnityEngine;

// Static utility class — no MonoBehaviour needed since it only does math.
// ClassroomManager calls these when generating the final report.
public static class ExamStats
{
    // Returns the arithmetic mean (average) of a list of integer scores.
    // scores.Average() returns a double, so we cast to float for Unity's math.
    public static float CalculateMean(List<int> scores)
    {
        return (float)scores.Average();
    }

    // Returns the median score — the middle value when the list is sorted.
    // For an even-length list we average the two middle elements.
    public static float CalculateMedian(List<int> scores)
    {
        var sorted = scores.OrderBy(n => n).ToList();
        int count = sorted.Count;

        if (count == 0)
        {
            return 0;
        }

        if (count % 2 == 0)
        {
            // Even number of scores: average the two middle ones
            return (sorted[count / 2 - 1] + sorted[count / 2]) / 2f;
        }

        // Odd number of scores: return the exact middle element
        return sorted[count / 2];
    }

    // Returns the sample standard deviation — how spread out the scores are.
    // We divide by (n - 1) rather than n because this is sample data, not the full population.
    public static float CalculateStdDev(List<int> scores)
    {
        // Standard deviation is undefined for fewer than two data points
        if (scores.Count <= 1)
        {
            return 0;
        }

        float avg = CalculateMean(scores);

        // Sum the squared differences from the mean
        float sum = scores.Sum(d => Mathf.Pow(d - avg, 2));

        // Divide by (n - 1) for sample variance, then take the square root
        return Mathf.Sqrt(sum / (scores.Count - 1));
    }
}
