using Microsoft.ML.Data;

namespace SpamDetection.Models {
    public class SpamClass {
        [LoadColumn(0)] public string Label { get; set; }
        [LoadColumn(1)] public string Message { get; set; } 
    }
}