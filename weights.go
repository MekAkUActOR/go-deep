package deep

import "math/rand"

// A WeightInitializer returns a (random) weight
type WeightInitializer func() float64

// NewUniform returns a uniform weight generator
func NewUniform(stdDev, mean float64, randg *rand.Rand) WeightInitializer {
	return func() float64 { return Uniform(stdDev, mean, randg) }
}

// Uniform samples a value from u(mean-stdDev/2,mean+stdDev/2)
func Uniform(stdDev, mean float64, randg *rand.Rand) float64 {
	return (randg.Float64()-0.5)*stdDev + mean

}

// NewNormal returns a normal weight generator
func NewNormal(stdDev, mean float64, randg *rand.Rand) WeightInitializer {
	return func() float64 { return Normal(stdDev, mean, randg) }
}

// Normal samples a value from N(μ, σ)
func Normal(stdDev, mean float64, randg *rand.Rand) float64 {
	return randg.NormFloat64()*stdDev + mean
}

// NewSetPara returns a scalar generator
func NewSetPara(stdDev, mean float64, randg *rand.Rand) WeightInitializer {
	return func() float64 { return SetPara(stdDev, mean, randg) }
}

// SetPara returns a scalar
func SetPara(stdDev, mean float64, randg *rand.Rand) float64 {
	return stdDev
}
