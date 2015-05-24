using System;
using System.Collections.Generic;

namespace DatasetUtility
{
    public class Fold
    {
        private List<Sample> samples;

        public Fold(int numOfSamples)
        {
            samples = new List<Sample>(numOfSamples);
        }

        public void Add(Sample sample)
        {
            if (samples.Count < samples.Capacity)
                samples.Add(sample);
        }

        public void Clear()
        {
            samples.Clear();
        }

        public Fold Clone()
        {
            Fold clone = new Fold(samples.Capacity);
            clone.samples = new List<Sample>(samples);
            return clone;
        }

        public int Count
        {
            get { return samples.Count; }
        }

        public void Shuffle()
        {
            Random rng = new Random();
            int n = samples.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                Sample value = samples[k];
                samples[k] = samples[n];
                samples[n] = value;
            }
        }
    }
}
