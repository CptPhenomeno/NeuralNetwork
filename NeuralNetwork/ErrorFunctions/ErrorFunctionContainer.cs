namespace NeuralNetwork.ErrorFunctions
{
    using System;

    using MathNet.Numerics;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;    

    public class ErrorFunctionContainer
    {

        public static ErrorFunction SQUARED_ERROR
        {
            get
            {
                return (targetValues, outputValues) =>
                {
                    return Distance.SSD(targetValues, outputValues) / 2.0;
                };
            }
        }

        public static ErrorFunction EUCLIDEAN_ERROR
        {
            get
            {
                return (targetValues, outputValues) =>
                {
                    return Distance.Euclidean(targetValues, outputValues);
                };
            }
        }

    }
}
