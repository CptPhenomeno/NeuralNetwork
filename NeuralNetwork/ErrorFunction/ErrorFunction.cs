namespace NeuralNetwork.ErrorFunction
{
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class ErrorFunction
    {
        public static double MSE(Vector<double> targetValues, Vector<double> outputValues)
        {
            Vector<double> errorVector = targetValues - outputValues;

            return errorVector.DotProduct(errorVector);
        }

    }
}
