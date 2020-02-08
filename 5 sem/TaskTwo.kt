import kotlin.math.absoluteValue
import kotlin.math.pow
import kotlin.math.sign

private val a: DoubleArray  =  doubleArrayOf(1.0,
                                            -10000.003,
                                            -999_999_98.999_998,
                                             100_000_029_0000.017,
                                            -999_999_99.999_998,
                                             100_000_030_0000.02)

val t = DoubleArray(a.size) { i -> a[i] / 100_000_030_0000.02 }

fun func(x: Double): Double = a[0] * x.pow(5) + a[1] * x.pow(4) + a[2] * x.pow(3) + a[3] * x.pow(2) + a[4] * x + a[5]
fun func1(x: Double): Double = t[0] * x.pow(5) + t[1] * x.pow(4) + t[2] * x.pow(3) + t[3] * x.pow(2) + t[4] * x + t[5]

fun main() {

    val bA = DoubleArray(a.size) { i -> a[i].absoluteValue }.drop(1).max()!!
    val bB = DoubleArray(a.size) { i -> a[i].absoluteValue }.dropLast(1).max()!!
    val bLeft = a.last().absoluteValue / (a.last().absoluteValue + bB)
    val bRight = 1 + bA / a[0].absoluteValue

    // 3 перемены знака
    var nulls = 0
    //println(bLeft)
    //println(bRight)

    val n = 1000
    val delta = 0.09 / n

    for (i in 1..n) {
        //print(func(10000.0 + (i - 1) * delta))
        //println(func1(10000.0 + (i - 1) * delta))
        if (nulls < 3) {
            if (sign(func(10000.0 + (i - 1) * delta)) != sign(func(10000.0 + i * delta))) {
                print("[${10000.0 + (i - 1) * delta}, ${10000.0 + i * delta}]")
                println("  ${func(10000.0 + (i - 1) * delta)}   ${func(10000.0 + i * delta)}")
                //nulls++
            }
            if (sign(func(-10000.001 + (i - 1) * delta)) != sign(func(-10000.001 + i * delta))) {
                print("[${-10000.001 + (i - 1) * delta}, ${-10000.001 + i * delta}]")
                println("  ${func(-10000.001 + (i - 1) * delta)}   ${func(-10000.001 + i * delta)}")
                //nulls++
            }
            /*
            if (sign(func(-bLeft - delta * (i - 1))) != sign(func(-bLeft - delta * i))) {
                println("${-bLeft - delta * (i - 1)} ${-bLeft - delta * i}")
                nulls++
            }*/
        } else break
    }

}
