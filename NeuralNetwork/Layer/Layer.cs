namespace NeuralNetwork.Layer
{
    using System;

    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.Distributions;

    using NeuralNetwork.Activation;    
    /*
    delegate double ActivationFunction(double x);
    delegate double ActivationFunctionDerivative(double x);
    */
    
    public class Layer
    {
        private int numberOfNeurons;
        private int numberOfInputs;
        private IActivationFunction activationFunction;
        private Matrix<double> weights;
        private Vector<double> bias;
        private Vector<double> localField;
        private Vector<double> output;

        private const double stdNormal = 1;
        

        public Layer(IActivationFunction activationFunction, int numOfNeurons, int numOfInputs)
        {
            this.numberOfNeurons = numOfNeurons;
            this.numberOfInputs = numOfInputs;
            this.activationFunction = activationFunction;
            bias = Vector<double>.Build.Random(NumberOfNeurons, new Normal(0, stdNormal));
            localField = Vector<double>.Build.Dense(NumberOfNeurons);
            output = Vector<double>.Build.Dense(NumberOfNeurons);
            weights = Matrix<double>.Build.Random(NumberOfNeurons, NumberOfInputs, new Normal(0, stdNormal));
            
        }

        public void ComputeOutput(Vector<double> input)
        {
            //Compute the local field without the bias
            weights.Multiply(input, localField);
            //Add the bias for the real induced local field
            localField.Add(bias, localField);
            //Apply the activation function to the local field
            localField.Map((l => activationFunction.Function(l)), output);
        }

        public void Update(Matrix<double> weightsUpdates, Vector<double> biasesUpdates)
        {
            weights.Add(weightsUpdates, weights);
            bias.Add(biasesUpdates, bias);
        }

        #region Getter & Setter

        public IActivationFunction ActivationFunction 
        { 
            get { return activationFunction; } 
        }

        public Matrix<double> Weights
        {
            get { return weights; }
            set { weights = value; }
        }

        public Vector<double> Bias
        {
            get { return bias; }
            set { bias = value; }
        }

        public Vector<double> Output
        {
            get { return output; }
        }

        public Vector<double> LocalFieldDifferentiated
        {
            get { return localField.Map((l => activationFunction.Derivative(l))); }
        }

        public int NumberOfNeurons
        {
            get { return numberOfNeurons; }
        }

        public int NumberOfInputs
        {
            get { return numberOfInputs; }
        }

        #endregion

        // override object.Equals
        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
            {
                return false;
            }

            Layer otherLayer = (Layer)obj;

            if (NumberOfNeurons                 == otherLayer.NumberOfNeurons &&
                NumberOfInputs                  == otherLayer.numberOfInputs  &&
                ActivationFunction.GetType()    == otherLayer.ActivationFunction.GetType())
            {
                bool equality = true;

                equality &= Weights.Equals(otherLayer.Weights);
                equality &= Bias.Equals(otherLayer.Bias);

                return equality;
            }

            return false;
        }

        public void RandomizeWeights()
        {
            bias.Clear();
            bias = null;
            bias = Vector<double>.Build.Random(NumberOfNeurons, new Normal(0, stdNormal));

            weights.Clear();
            weights = null;
            weights = Matrix<double>.Build.Random(NumberOfNeurons, NumberOfInputs, new Normal(0, stdNormal));
        }
    }
}
