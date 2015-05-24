using System.Collections.Generic;

namespace DatasetUtility
{
    public class Dataset
    {
        private readonly int SAMPLES_PER_FOLD;

        private List<Fold> trainingFolds;
        private Fold validationFold;

        private Fold temporaryFold;

        public Dataset (int foldNumber, int samplesPerFold, int numOfInput, int numOfOutput)
        {
            trainingFolds = new List<Fold>(foldNumber - 1);
            SAMPLES_PER_FOLD = samplesPerFold;
        }

        public void AddSampleToDataset(Sample sample)
        {
            if (temporaryFold == null)
                temporaryFold = new Fold(SAMPLES_PER_FOLD);

            if (temporaryFold.Count < SAMPLES_PER_FOLD)
                temporaryFold.Add(sample);

            if (temporaryFold.Count == SAMPLES_PER_FOLD)
            {
                if (validationFold == null)
                    validationFold = temporaryFold.Clone();
                else if (trainingFolds.Count < trainingFolds.Capacity)
                    trainingFolds.Add(temporaryFold.Clone());

                temporaryFold.Clear();
            }
        }

        public void ChangeValidationFold()
        {
            trainingFolds.Add(validationFold);
            validationFold = trainingFolds[0];
            trainingFolds.RemoveAt(0);
        }

        public void Shuffle()
        {
            validationFold.Shuffle();
            foreach (Fold f in trainingFolds)
                f.Shuffle();
        }

        public List<Fold> TrainingFold { get { return trainingFolds; } }

        public Fold ValidationFold { get { return validationFold; } }

    }
}
