import kotlin.math.PI
import kotlin.math.pow
import kotlin.math.sqrt

const val y = PI / 999   /// = d / (a + b + c)
private const val a = 99
private const val b = 500
const val c = 400

val q1 = (b / 2 + sqrt((b * b / 4 - a * c).toDouble())) / a
val q2 = (b / 2 - sqrt((b * b / 4 - a * c).toDouble())) / a

var y0 = y   /// alpha
var y1 = y + 0.000_000_000_000_001 /// beta

val c1 = (y1 - y + (y - y0) * q2) / (q1 - q2)
val c2 = (y1 - y + (y - y0) * q1) / (q2 - q1)


fun main() {
    var yn = 0.0

    println("q1 = $q1  q2 = $q2")
    println("c1 = $c1  c2 = $c2")
    val n = readLine()!!.toInt()

    if (n in 0..1) println("y($n) = $y") else {
        for (i in 2..n) {
            yn = (999 - y1 * 999 / PI * b - y0 * 999 / PI * c) / a * y
            y0 = y1
            y1 = yn
            println("y($i) = $yn    ${c1 * q1.pow(i) + c2 * q2.pow(i) + y}")
        }
    }
}