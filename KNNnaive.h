#pragma once
#include "vecoperation.h"

class KNNnaive
{
	private:
		vecop::features *__X;
		vecop::class_label *__Y;
		int __k;

	public:
		KNNnaive(int k = 3);
		
		void fit(const vecop::features &X,const vecop::class_label &Y);
		int predict(const vecop::feature &X);
};

