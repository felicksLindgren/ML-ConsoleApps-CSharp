using Microsoft.ML.Data;
using System.Linq;

namespace SpamDetection.Models
{
    public class ProcessedData
    {
        public string Label { get; set; }
        public VBuffer<float> Features { get; set; }
        public float[] GetFeatures() => (float[])Features.DenseValues().ToArray();
        public float GetLabel() => Label == "spam" ? 1.0f : 0.0f;
    }
}