import koma.emul
import koma.end
import koma.mat
import koma.matrix.Matrix

/**
 * Neural network of type Perceptron being used to predict binary OR operations.
 */
class Perceptron(input: Matrix<Double> = mat[0, 0 end 1, 0 end 0, 1 end 1, 1],
                 output: Matrix<Double> = mat[0, 1, 1, 1].T) {


    // Maximum number of epochs
    val epochMax = 30
    // Counter for epochs
    var epochCount = 0
    val learningTax = 0.3
    val input: Matrix<Double>
    val output: Matrix<Double>
    // Synaptic weights for: entries 1, entries 2 and bias
    var weights: Matrix<Double>
    val bias: Matrix<Double>

    init {
        // Zeroed matrix with bias
        this.input =
            mat[0, 0, 1 end
                0, 0, 1 end
                0, 0, 1 end
                0, 0, 1]

        bias = mat[0, 0, 1]

        // Expected values from OR over above entries matrix
        this.output = output

        // Weights for first entry, second entry and bias
        weights = mat[0, 0, 0]

        this.input.set(0..input.numRows() - 1, 0..input.numCols() - 1, input)
    }

    fun predict(row: Matrix<Double>): Int {
        bias.set(0, 0..1, row)
        // Activation signal
        val u = (weights emul bias).elementSum()
        // Neuron output signal
        val y = activationFunction(u)
        return y
    }

    fun train() {
        println("EPOCH: $epochCount")
        var err = false
        var rowIndex = 0
        input.forEachRow { row ->
            val y = predict(row)
            val error = output[rowIndex, 0] - y
            rowIndex++
            if (error != 0.0) {
                err = true
                println("Sample ${rowIndex + 1}:$row -> error = $error")
                adjustWeights(row, error)
            }
        }

        epochCount++
        if (epochCount < epochMax && err == true) {
            train()
        }
    }

    /**
     * Activation function used is binary step function. [u] is the activation signal.
     */
    private fun activationFunction(u: Double): Int {
        return if (u >= 0) 1 else 0
    }

    private fun adjustWeights(row: Matrix<Double>, error: Double) {
        weights = weights.mapIndexed { _, col, w ->
            w + learningTax * error * row[col]
        }
        println("Weights adjusted to: $weights")
    }
}