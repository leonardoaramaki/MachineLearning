import koma.*

fun main(args: Array<String>) {
    val nn = Perceptron(
            input = mat[0, 0 end 1, 0 end 0, 1 end 1, 1],
            output = mat[0, 1, 1, 1].T
    )

    nn.train()

    println(nn.predict(mat[0, 0]))
    println(nn.predict(mat[1, 0]))
    println(nn.predict(mat[0, 1]))
    println(nn.predict(mat[1, 1]))
}