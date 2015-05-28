using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace DatasetUtility
{
    public class Sample
    {
        private Vector<double> input;
        private Vector<double> output;

        public Sample(Vector<double> input, Vector<double> output)
        {
            this.input = input;
            this.output = output;
        }

        public Sample(double[] input, double[] output)
        {
            Vector<double> inputVector = Vector<double>.Build.DenseOfArray(input);
            Vector<double> outputVector = Vector<double>.Build.DenseOfArray(output);

            this.input = inputVector;
            this.output = outputVector;
        }

        public Vector<double> Input { get { return input; } }

        public Vector<double> Output { get { return output; } }
    }
}
