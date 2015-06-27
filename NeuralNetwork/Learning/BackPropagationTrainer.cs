namespace NeuralNetwork.Learning
{
    using System;
    using System.Collections.Concurrent;

    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;
    
    
    using NeuralNetwork.Network;
    using NeuralNetwork.Utils.Extensions;

    using DatasetUtility;
    using System.Diagnostics;
using System.IO;
    
    public class BackPropagationTrainer
    {
        private NeuralNet net;
        private Backpropagation backpropagation;
        private BlockingCollection<string> errorOnEpochs;
        private bool logInformationEnabled;
        
        private double maxError;
        private int maxEpoch;
        private int numFold;
        private int batchSize;

        private bool running;

        public BackPropagationTrainer(NeuralNet net, double learningRate = 0.3, double momentum = 0.0, 
                                      double weightDecay = 0.001, double maxError = 0.01, int maxEpoch = 1000, 
                                      int numFold = 4, int batchSize = 1)
        {
            this.net = net;
            backpropagation = new Backpropagation(net, learningRate, momentum, weightDecay, batchSize);
            logInformationEnabled = false;
            running = false;
            this.maxError = maxError;
            this.maxEpoch = maxEpoch;
            this.numFold = numFold;
            this.batchSize = batchSize;
        }

        public NeuralNet Learn(Dataset trainSet, string filepath = null, Dataset testSet = null)
        {
            running = true;
            double trainingError = 0.0;
            double testError = 0.0;
            int epoch = -1;
            int numOfElemInEpoch = trainSet.Size;

            StreamWriter writer = null;

            if (filepath != null)
                writer = new StreamWriter(filepath, false);

            do
            {
                ++epoch;

                if (testSet != null)
                {
                    testError = RunTestSet(testSet);
                }

                if (epoch % 100 == 0)
                    Console.WriteLine("--- Epoch {0} ---", epoch);
                trainingError = RunEpoch(trainSet);

                if (writer != null)
                {
                    if (testSet != null)
                        writer.WriteLine("{0} {1}", trainingError.ToString(System.Globalization.CultureInfo.InvariantCulture),
                                                testError.ToString(System.Globalization.CultureInfo.InvariantCulture));
                    else
                        writer.WriteLine(trainingError.ToString(System.Globalization.CultureInfo.InvariantCulture));
                }

                //Shuffle the examples
                trainSet.Shuffle();

            } while (trainingError > MaxError && epoch < MaxEpoch);

            if (writer != null)
                writer.Close();

            return net;
        }

        private NeuralNet LearnWithStop(Dataset trainSet, double stopValue, string filepath = null, Dataset testSet = null)
        {
            running = true;
            double trainingError = 0.0;
            double testError = 0.0;
            int epoch = -1;
            int numOfElemInEpoch = trainSet.Size;
            StreamWriter writer = null;

            if (filepath != null)
                writer = new StreamWriter(filepath, false);

            do
            {
                ++epoch;

                if (testSet != null)
                {
                    testError = RunTestSet(testSet);
                }

                //if (epoch % 100 == 0)
                //    Console.WriteLine("--- Epoch {0} ---", epoch);
                trainingError = RunEpoch(trainSet);

                if (writer != null)
                {
                    if (testSet != null)
                        writer.WriteLine("{0} {1}", trainingError.ToString(System.Globalization.CultureInfo.InvariantCulture),
                                                testError.ToString(System.Globalization.CultureInfo.InvariantCulture));
                    else
                        writer.WriteLine(trainingError.ToString(System.Globalization.CultureInfo.InvariantCulture));
                }

                //Shuffle the examples
                trainSet.Shuffle();

            } while (trainingError > stopValue && epoch < MaxEpoch);

            if (writer != null)
                writer.Close();

            return net;
        }

        private double[] KFoldsTrain(Dataset trainSet, int folds)
        {
            double trainingError = 0.0;
            double validationError = 0.0;

            int trainSetSize = trainSet.Size;
            int foldSize = trainSetSize / folds;
            int lastFoldSize = foldSize + (trainSetSize % folds);

            int oldBatchSize = BatchSize;

            for (int k = 0; k < folds; k++)
            {
                int validationSize = (k == folds - 1) ? lastFoldSize : foldSize;
                int trainingSize = trainSetSize - validationSize;

                double foldValidationError = 0;
                double foldTrainingError = 0;

                BatchSize = trainingSize;

                //Train
                int samplesUsed = 0;
                for (int trainIndex = (k * foldSize + validationSize) % trainSetSize;
                     trainIndex != k * foldSize;
                     trainIndex = ((trainIndex + 1) % trainSetSize))
                {
                    foldTrainingError += backpropagation.Run(trainSet[trainIndex]);

                    if (double.IsNaN(foldTrainingError))
                        Console.WriteLine("Train error is NaN");
                    if (double.IsInfinity(foldTrainingError))
                        Console.WriteLine("Train error is Infinity");

                    ++samplesUsed;

                    if (samplesUsed % BatchSize == 0)
                        backpropagation.UpdateNetwork();
                }

                foldTrainingError /= trainingSize;

                //Cross Validation
                for (int valIndex = k * foldSize; valIndex < k * foldSize + validationSize; valIndex++)
                {
                    Sample validationSample = trainSet[valIndex];
                    net.ComputeOutput(validationSample.Input);
                    Vector<double> netError = validationSample.Output - net.Output;
                    foldValidationError += netError.DotProduct(netError);
                    foldValidationError /= 2;
                }

                foldValidationError /= validationSize;
                validationError += foldValidationError;
                trainingError += foldTrainingError;
            }

            validationError /= folds;
            trainingError /= folds;

            BatchSize = oldBatchSize;

            return new double[] {validationError, trainingError};
        }

        private double[] CrossValidation(Dataset trainSet, int folds = 4, int maxFails = 10)
        {
            double validationError = 0.0;
            double trainingError = 0.0;

            //double previousError = double.MaxValue;

            double bestValidationError = double.MaxValue;
            double bestTrainingError = double.MaxValue;

            int validationFail = 0;

            int epoch = -1;
            
            do
            {
                ++epoch;

                trainSet.Shuffle();

                double[] errors = KFoldsTrain(trainSet, folds);
                
                validationError = errors[0];
                trainingError = errors[1];

                if (validationError < bestValidationError)
                {
                    bestValidationError = validationError;
                    bestTrainingError = trainingError;
                    validationFail = 0;
                }
                else
                {
                    validationFail++;
                }

                //if (validationError <= previousError)
                //{
                //    validationFail = 0;
                //    if (validationError < bestValidationError)
                //    {
                //        bestValidationError = validationError;
                //        bestTrainingError = trainingError;
                //    }
                //}
                //else
                //    validationFail++;

                //previousError = validationError;

            } while (validationFail < maxFails && bestValidationError > MaxError && epoch < MaxEpoch);

            return new double[] {bestValidationError, bestTrainingError};
        }

        public void CrossValidationLearn(Dataset trainSet, int folds = 5, string filepath = null)
        {
            double validationError = 0.0;
            double trainingError = 0.0;

            int epoch = -1;

            StreamWriter writer = null;

            if (filepath != null)
                writer = new StreamWriter(filepath, false);

            do
            {
                ++epoch;

                if (epoch % 100 == 0)
                    Console.WriteLine("-- Epoch {0} --", epoch);

                double[] errors = KFoldsTrain(trainSet, folds);

                validationError = errors[0];
                trainingError = errors[1];

                if (writer != null)
                {
                    writer.WriteLine("{0},{1}", trainingError.ToString(System.Globalization.CultureInfo.InvariantCulture),
                                                validationError.ToString(System.Globalization.CultureInfo.InvariantCulture));
                }

                trainSet.Shuffle();

            } while (trainingError > MaxError && epoch < MaxEpoch);

            if (writer != null)
                writer.Close();
        }

        public NeuralNet CrossValidationLearnWithModelSelection(Dataset trainSet,
                    double[] etaValues, double[] alphaValues,  double[] lambdaValues,
                    int folds = 4, int maxFails = 10, string filepath = null, Dataset testSet = null)
        {
            double bestValidationError = double.MaxValue;
            double bestTrainingError = double.MaxValue;

            double bestEta = 0;
            double bestAlpha = 0;
            double bestLambda = 0;

            int numOfTries = 1;

            NeuralNet startNet = NetworkFactory.Clone(net);
            
            //Hyperparameter grid search
            foreach (double eta in etaValues)
            {
                foreach (double alpha in alphaValues)
                {
                    foreach (double lambda in lambdaValues)
                    {
                        Console.WriteLine("Train network with eta {0}, alpha {1} and lambda {2}", eta, alpha, lambda);

                        LearningRate = eta;
                        Momentum = alpha;
                        WeightDecay = lambda;

                        double bestValidationErrorMean = 0;
                        double bestTrainingErrorMean = 0;

                        for (int t = 0; t < numOfTries; t++)
                        {
                            //Console.WriteLine("Test {0}", t + 1);
                            //Test network generalization
                            double[] errors = CrossValidation(trainSet, folds, maxFails);
                            double validationError = errors[0];
                            double trainingError = errors[1];

                            bestValidationErrorMean += validationError;
                            bestTrainingErrorMean += trainingError;

                            net.RandomizeWeights();
                            backpropagation.Net = net;
                        }

                        bestValidationErrorMean /= numOfTries;
                        bestTrainingErrorMean /= numOfTries;

                        if (!(double.IsNaN(bestValidationErrorMean)) && bestValidationErrorMean < bestValidationError)
                        {
                            bestValidationError = bestValidationErrorMean;
                            bestTrainingError = bestTrainingErrorMean;
                            
                            bestEta = eta;
                            bestAlpha = alpha;
                            bestLambda = lambda;
                        }

                        net = null;
                        net = NetworkFactory.Clone(startNet);
                        backpropagation.Net = net;
                    }
                }
            }

            Console.WriteLine("Best eta => {0} Best alpha {1} Best lambda {2}", bestEta, bestAlpha, bestLambda);

            LearningRate = bestEta;
            Momentum = bestAlpha;
            WeightDecay = bestLambda;

            return LearnWithStop(trainSet, bestTrainingError, filepath, testSet);            
        }

        private double RunEpoch(Dataset trainSet)
        {
            double error = 0.0;
            int sizeOfTrainingSet = trainSet.Size;
            int next;

            for (next = 0; next < sizeOfTrainingSet; next++)
            {
                error += backpropagation.Run(trainSet[next]);
                
                //Update the network only after batchSize examples
                if ((next + 1) % BatchSize == 0)
                    backpropagation.UpdateNetwork();
            }

            if (next % BatchSize != 0)
                backpropagation.UpdateNetwork();

            error /= sizeOfTrainingSet;

            return error;
        }

        private double RunTestSet(Dataset testSet)
        {
            double error = 0.0;
            int sizeOfTestSet = testSet.Size;

            for (int next = 0; next < sizeOfTestSet; next++)
            {
                Sample sample = testSet[next];
                net.ComputeOutput(sample.Input);
                //Vector with error for each output of the network
                Vector<double> netError = sample.Output - net.Output;
                //I think that this matrix is 1x1 but is better check...
                error += netError.DotProduct(netError);
            }

            error /= sizeOfTestSet;

            return error;
        }

        #region Getter & Setter

        public double MaxError
        {
            get { return maxError; }
            set { maxError = value; }
        }

        public int MaxEpoch
        {
            get { return maxEpoch; }
            set { maxEpoch = value; }
        }

        public int NumFold
        {
            get { return numFold; }
            set { numFold = value; }
        }

        public int BatchSize
        {
            get { return batchSize; }
            set { batchSize = value; }
        }

        public bool LogInformationEnabled
        {
            get { return logInformationEnabled; }
            set 
            {
                if (!running)
                    logInformationEnabled = value;
                else
                    Console.WriteLine("Cannot enable/disable log when the learning run!");
            }
        }

        public void EnableLogging(BlockingCollection<string> logStorage)
        {
            LogInformationEnabled = true;
            errorOnEpochs = logStorage;
        }

        public void DisableLogging()
        {
            LogInformationEnabled = false;
        }

        #endregion

        #region BackPropagation Getter & Setter

        public double LearningRate
        {
            get { return backpropagation.LearningRate; }
            set { backpropagation.LearningRate = value; }
        }

        public double Momentum
        {
            get { return backpropagation.Momentum; }
            set { backpropagation.Momentum = value; }
        }

        public double WeightDecay
        {
            get { return backpropagation.WeightDecay; }
            set { backpropagation.WeightDecay = value; }
        }

        #endregion

    }
}
