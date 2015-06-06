namespace NeuralNetwork.Learning
{
    using System;
    using System.Collections.Concurrent;

    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;
    
    
    using NeuralNetwork.Network;
    using NeuralNetwork.Utils.Extensions;

    using DatasetUtility;
    
    public class BackPropagationTrainer
    {
        private NeuralNet net;
        private BackPropagation backpropagation;
        private BlockingCollection<string> errorOnEpochs;
        private bool logInformationEnabled;
        
        private double maxError;
        private int maxEpoch;
        private int numFold;
        private int batchSize;

        private bool running;

        public BackPropagationTrainer(NeuralNet net, double learningRate = 0.3, double momentum = 0.0, double maxError = 0.01,
                                      int maxEpoch = 1000, int numFold = 4, int batchSize = 1)
        {
            this.net = net;
            backpropagation = new BackPropagation(net, learningRate, momentum, batchSize);
            logInformationEnabled = false;
            running = false;
            this.maxError = maxError;
            this.maxEpoch = maxEpoch;
            this.numFold = numFold;
            this.batchSize = batchSize;
        }

        public NeuralNet Learn(Dataset trainSet, Dataset testSet = null)
        {
            running = true;
            double trainingError = 0.0;
            double testError = 0.0;
            int epoch = -1;
            int numOfElemInEpoch = trainSet.Size;

            do
            {
                ++epoch;
                if (epoch % 100 == 0)
                    Console.WriteLine("--- Epoch {0} ---", epoch);
                trainingError = RunEpoch(trainSet);
                trainingError /= numOfElemInEpoch;
                if (LogInformationEnabled)
                {
                    string log = epoch + ":" + trainingError;

                    if (testSet != null && testSet != null)
                    {
                        testError = RunTestSet(testSet);
                        log += ":" + testError;
                    }

                    errorOnEpochs.Add(log);
                }

                //Shuffle the examples
                trainSet.Shuffle();

            } while (trainingError > MaxError && epoch < MaxEpoch);

            if (logInformationEnabled)
                errorOnEpochs.CompleteAdding();
            running = false;

            return net;
        }

        private double CrossValidate(Dataset trainSet, int folds = 4, int maxFails = 10)
        {
            NeuralNet savedNet = NetworkFactory.Clone(net);
            double trainingError = 0.0;
            double validationError = 0.0;
            double minValidationError = Double.MaxValue;

            int validationFail = 0;

            int epoch = -1;
            int savedEpoch = -1;
            int trainSetSize = trainSet.Size;

            int foldSize = trainSetSize / folds;
            int lastFoldSize = foldSize + (trainSetSize % folds);

            do
            {
                ++epoch;

                //if (epoch % 100 == 0)
                //    Console.WriteLine("--- Epoch {0} ---", epoch);

                trainSet.Shuffle();
                
                for (int k = 0; k < folds; k++)
                {
                    int validationSize = (k == folds - 1) ? lastFoldSize : foldSize;
                    int trainingSize = trainSetSize - validationSize;

                    double foldValidationError = 0;

                    //Train
                    for (int trainIndex = (k * foldSize + validationSize) % trainSetSize;
                         trainIndex != k * foldSize;
                         trainIndex = ((trainIndex + 1) % trainSetSize))
                    {
                        trainingError += backpropagation.Run(trainIndex, trainSet);
                        backpropagation.UpdateNetwork();
                    }

                    trainingError /= trainingSize;

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
                }

                validationError /= folds;

                if (validationError <= minValidationError)
                {
                    minValidationError = validationError;
                    validationFail = 0;
                    savedNet = NetworkFactory.Clone(net);
                    savedEpoch = epoch;
                }
                else
                {
                    validationFail++;
                    if (validationFail == maxFails)
                    {
                        net = null;
                        net = NetworkFactory.Clone(savedNet);
                    }
                }

            } while (validationFail < maxFails && validationError > MaxError && epoch < MaxEpoch);
            
            savedNet = null;
            
            //TODO
            return minValidationError;
        }

        public NeuralNet CrossValidationLearn(Dataset trainSet, int folds = 4, Dataset testSet = null)
        {
            NeuralNet savedNet = NetworkFactory.Clone(net);
            double trainingError = 0.0;
            double validationError = 0.0;
            double minValidationError = Double.MaxValue;
            //double testError = 0.0;
            
            int validationFail = 0;
            int maxFail = 10;
            
            int epoch = -1;
            int savedEpoch = -1;
            int trainSetSize = trainSet.Size;

            int foldSize = trainSetSize / folds;
            int lastFoldSize = foldSize + (trainSetSize % folds);

            do
            {
                ++epoch;

                //if (epoch % 100 == 0)
                    Console.WriteLine("--- Epoch {0} ---", epoch);

                for (int k = 0; k < folds; k++)
                {
                    int validationSize = (k == folds - 1) ? lastFoldSize : foldSize;
                    int trainingSize = trainSetSize - validationSize;

                    double foldValidationError = 0;

                    //Train
                    for (int trainIndex = (k * foldSize + validationSize) % trainSetSize;
                         trainIndex != k * foldSize;
                         trainIndex = ((trainIndex + 1) % trainSetSize))
                    {
                        trainingError += backpropagation.Run(trainIndex, trainSet);
                        backpropagation.UpdateNetwork();
                    }

                    trainingError /= trainingSize;

                    //Cross Validation
                    for (int valIndex = k * foldSize; valIndex < k * foldSize + validationSize; valIndex++)
                    {
                        Sample validationSample = trainSet[valIndex];
                        net.ComputeOutput(validationSample.Input);
                        Vector<double> netError = validationSample.Output - net.Output;
                        foldValidationError += netError.DotProduct(netError);
                    }

                    foldValidationError /= validationSize;
                    validationError += foldValidationError;
                }

                validationError /= folds;

                trainSet.Shuffle();

                if (validationError <= minValidationError)
                {
                    minValidationError = validationError;
                    validationFail = 0;
                    savedNet = NetworkFactory.Clone(net);
                    savedEpoch = epoch;
                }
                else
                {
                    validationFail++;
                    if (validationFail == maxFail)
                    {
                        Console.WriteLine("[{0}] -> Too much validation fails. Restore the net at epoch {1}", epoch, savedEpoch);
                        net = NetworkFactory.Clone(savedNet);
                        backpropagation.Net = net;
                    }
                        
                }

            } while (validationFail < maxFail && validationError > MaxError && epoch < MaxEpoch);

            
            Console.WriteLine("Training error: {0}",trainingError);
            Console.WriteLine("Validation error: {0}", validationError);

            return net;
        }

        public NeuralNet CrossValidationLearnWithModelSelection(Dataset trainSet,
                    double minEta, double etaStep, double maxEta, double minAlpha, double alphaStep, double maxAlpha, 
                    int folds = 4, int maxFails = 10, Dataset testSet = null)
        {
            //NeuralNet startNet = NetworkFactory.Clone(net);
            NeuralNet tmpNet = null;
            NeuralNet bestNet = null;
            double bestValidationError = Double.MaxValue;
            double bestEta = 0;
            double bestAlpha = 0;

            int numOfTries = 5;

            //Hyperparameter grid search
            for (double eta = minEta; eta <= maxEta; eta += etaStep)
            {
                for (double alpha = minAlpha; alpha <= maxAlpha; alpha += alphaStep)
                {
                    Console.WriteLine("Train network with eta {0} and alpha {1}", eta, alpha);

                    LearningRate = eta;
                    Momentum = alpha;

                    double meanValidationError = 0;
                    double bestNetError = Double.MaxValue;

                    for (int t = 0; t < numOfTries; t++)
                    {
                        //Test network generalization
                        double validationError = CrossValidate(trainSet, folds, maxFails);
                        meanValidationError += validationError;

                        if (!(Double.IsNaN(validationError)) && validationError < bestNetError)
                        {
                            tmpNet = NetworkFactory.Clone(net);
                        }

                        net.RandomizeWeights();
                        backpropagation.Net = net;
                    }

                    meanValidationError /= numOfTries;

                    if (!(Double.IsNaN(meanValidationError)) && meanValidationError < bestValidationError)
                    {
                        bestValidationError = meanValidationError;
                        bestEta = eta;
                        bestAlpha = alpha;
                        bestNet = null;
                        bestNet = NetworkFactory.Clone(tmpNet);
                    }

                    tmpNet = null;

                    net.RandomizeWeights();
                    backpropagation.Net = net;
                    //Train the start net (same weights)
                    //net = null;
                    //net = NetworkFactory.Clone(startNet);
                    //backpropagation.Net = net;
                }
            }

            net = null;
            //startNet = null;
            net = NetworkFactory.Clone(bestNet);
            backpropagation.Net = net;
            bestNet = null;

            Console.WriteLine("Best performance with eta {0} and alpha {1}", bestEta, bestAlpha);

            return net;
        }

        private double RunEpoch(Dataset trainSet)
        {
            double error = 0.0;
            int size = trainSet.Size;

            for (int next = 0; next < size; next += batchSize)
            {
                error += backpropagation.Run(next, trainSet);
                //Update the network only after batchSize examples
                backpropagation.UpdateNetwork();
            }

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

        #endregion
    }
}
