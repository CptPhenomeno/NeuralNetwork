using System;
using System.Collections.Generic;

namespace DatasetUtility
{
    public class Dataset
    {
        private List<Sample> samples;
        private int size;

        public Dataset ()
        {
            samples = new List<Sample>();
        }

        public void Add(Sample sample)
        {
            samples.Add(sample);
            size++;
        }

        public void Shuffle()
        {
            Random index = new Random(DateTime.Now.Millisecond);
            
            for (int i = 0; i < size - 1; i++)
            {
                int j = index.Next(i, size);
                Sample s = samples[i];
                samples[i] = samples[j];
                samples[j] = s;
            }
        }

        public List<Sample> Samples { get { return samples; } }

        public Sample this[int sampleIndex] { get { return samples[sampleIndex]; } }

        public int Size { get { return size; } }

    }
}
