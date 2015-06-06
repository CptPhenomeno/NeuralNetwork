namespace NeuralNetwork.Layer
{

    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra;
    
    using NeuralNetwork.Activation;    
  
    public class Layer
    {
        private int numberOfNeurons;
        private int numberOfInputs;
        private IActivationFunction activationFunction;
        private Matrix<double> weights;
        private Vector<double> bias;
        private Vector<double> localField;
        private Vector<double> output;

        private const double STD_NORMAL = 1;
        

        public Layer(IActivationFunction activationFunction, int numOfNeurons, int numOfInputs)
        {
            this.numberOfNeurons = numOfNeurons;
            this.numberOfInputs = numOfInputs;
            this.activationFunction = activationFunction;
            bias = Vector<double>.Build.Random(NumberOfNeurons, new Normal(0, STD_NORMAL));
            localField = Vector<double>.Build.Dense(NumberOfNeurons);
            output = Vector<double>.Build.Dense(NumberOfNeurons);
            weights = Matrix<double>.Build.Random(NumberOfNeurons, NumberOfInputs, new Normal(0, STD_NORMAL));
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

        public int NumberOfNeurons
        {
            get { return numberOfNeurons; }
        }

        public int NumberOfInputs
        {
            get { return numberOfInputs; }
        }
        
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

        #endregion

        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
            {
                return false;
            }

            Layer otherLayer = obj as Layer;

            return Equals(otherLayer);
        }

        private bool Equals(Layer l)
        {
            if (NumberOfNeurons == l.NumberOfNeurons &&
                NumberOfInputs == l.numberOfInputs &&
                ActivationFunction.GetType() == l.ActivationFunction.GetType())
            {
                bool equality = true;

                equality &= Weights.Equals(l.Weights);
                equality &= Bias.Equals(l.Bias);

                return equality;
            }

            return false;
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public void RandomizeWeights()
        {
            bias.Clear();
            bias = null;
            bias = Vector<double>.Build.Random(NumberOfNeurons, new Normal(0, STD_NORMAL));

            weights.Clear();
            weights = null;
            weights = Matrix<double>.Build.Random(NumberOfNeurons, NumberOfInputs, new Normal(0, STD_NORMAL));
        }
    }
}
