import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow

const val error = 0.000001

fun nFormula(n: Double): Double = n - (exp(1 - n.pow(2)) - n.pow(2)) / (-2 * n * exp(1 - n.pow(2)) - 2 * n)
fun secDer(n: Double): Double = (4 * n.pow(2) - 2) * exp(1 - n.pow(2)) - 2

fun main() {
    val m = 4
    var x0 = -2.0
    println("f(x0) = ${exp(1 - x0.pow(2)) - x0.pow(2)}  f''(x) = ${secDer(x0)}")

    var x1 = nFormula(x0)
    println(x1)

    while (abs(secDer(x1)) / 2 / m * (x0 - x1).pow(2) > error) {
        x0 = x1
        x1 = nFormula(x1)
        println(x1)
    }
}