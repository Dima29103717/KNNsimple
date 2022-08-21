#include "KNNnaive.h"
using namespace knn;
KNNnaive::KNNnaive(int k, DistMethod method) : __k(k), __method(method)
{
	this->__data = new __data_t();
}

KNNnaive::~KNNnaive()
{
	delete __data;
}

void KNNnaive::fit(const vecop::features& X, const vecop::class_label& Y)
{
	this->__data->__X = X;
	this->__data->__Y = Y;
	this->__data->dis.clear();
}

int KNNnaive::predict(const vecop::feature& X)
{
	return 0;
}
