#ifndef CZY_MATH_FIT
#define CZY_MATH_FIT
#include <vector>
/*
code by chen zhong yuan ����Զ����2014.03.20
homepage ��ҳ��http://blog.csdn.net/czyt1988/article/details/21743595
ref��http://blog.csdn.net/maozefa/article/details/1725535
*/
namespace czy {
	///
	/// \brief ���������
	///
	class Fit {
		std::vector<double> factor; ///<��Ϻ�ķ���ϵ��
		double ssr;                 ///<�ع�ƽ����
		double sse;                 ///<(ʣ��ƽ����)
		double rmse;                ///<RMSE���������
		std::vector<double> fitedYs;///<�����Ϻ��yֵ�������ʱ������Ϊ�������ʡ�ڴ�
	public:
		Fit() :ssr(0), sse(0), rmse(0) { factor.resize(2, 0); }
		~Fit() {}
		///
		/// \brief ֱ�����-һԪ�ع�,��ϵĽ������ʹ��getFactor��ȡ������ʹ��getSlope��ȡб�ʣ�getIntercept��ȡ�ؾ�
		/// \param x �۲�ֵ��x
		/// \param y �۲�ֵ��y
		/// \param isSaveFitYs ��Ϻ�������Ƿ񱣴棬Ĭ�Ϸ�
		///
		template<typename T>
		bool linearFit(const std::vector<typename T>& x, const std::vector<typename T>& y, bool isSaveFitYs = false)
		{
			return linearFit(&x[0], &y[0], getSeriesLength(x, y), isSaveFitYs);
		}
		template<typename T>
		bool linearFit(const T* x, const T* y, size_t length, bool isSaveFitYs = false)
		{
			factor.resize(2, 0);
			typename T t1 = 0, t2 = 0, t3 = 0, t4 = 0;
			for (int i = 0; i<length; ++i)
			{
				t1 += x[i] * x[i];
				t2 += x[i];
				t3 += x[i] * y[i];
				t4 += y[i];
			}
			factor[1] = (t3*length - t2*t4) / (t1*length - t2*t2);
			factor[0] = (t1*t4 - t2*t3) / (t1*length - t2*t2);
			//////////////////////////////////////////////////////////////////////////
			//�������
			calcError(x, y, length, this->ssr, this->sse, this->rmse, isSaveFitYs);
			return true;
		}
		///
		/// \brief ����ʽ��ϣ����y=a0+a1*x+a2*x^2+����+apoly_n*x^poly_n
		/// \param x �۲�ֵ��x
		/// \param y �۲�ֵ��y
		/// \param poly_n ������ϵĽ�������poly_n=2����y=a0+a1*x+a2*x^2
		/// \param isSaveFitYs ��Ϻ�������Ƿ񱣴棬Ĭ����
		/// 
		template<typename T>
		void polyfit(const std::vector<typename T>& x
			, const std::vector<typename T>& y
			, int poly_n
			, bool isSaveFitYs = true)
		{
			polyfit(&x[0], &y[0], getSeriesLength(x, y), poly_n, isSaveFitYs);
		}
		template<typename T>
		void polyfit(const T* x, const T* y, size_t length, int poly_n, bool isSaveFitYs = true)
		{
			factor.resize(poly_n + 1, 0);
			int i, j;
			//double *tempx,*tempy,*sumxx,*sumxy,*ata;
			std::vector<double> tempx(length, 1.0);

			std::vector<double> tempy(y, y + length);

			std::vector<double> sumxx(poly_n * 2 + 1);
			std::vector<double> ata((poly_n + 1)*(poly_n + 1));
			std::vector<double> sumxy(poly_n + 1);
			for (i = 0; i<2 * poly_n + 1; i++) {
				for (sumxx[i] = 0, j = 0; j<length; j++)
				{
					sumxx[i] += tempx[j];
					tempx[j] *= x[j];
				}
			}
			for (i = 0; i<poly_n + 1; i++) {
				for (sumxy[i] = 0, j = 0; j<length; j++)
				{
					sumxy[i] += tempy[j];
					tempy[j] *= x[j];
				}
			}
			for (i = 0; i<poly_n + 1; i++)
				for (j = 0; j<poly_n + 1; j++)
					ata[i*(poly_n + 1) + j] = sumxx[i + j];
			gauss_solve(poly_n + 1, ata, factor, sumxy);
			//������Ϻ�����ݲ��������
			fitedYs.reserve(length);
			calcError(&x[0], &y[0], length, this->ssr, this->sse, this->rmse, isSaveFitYs);

		}
		/// 
		/// \brief ��ȡϵ��
		/// \param ���ϵ��������
		///
		void getFactor(std::vector<double>& factor) { factor = this->factor; }
		/// 
		/// \brief ��ȡ��Ϸ��̶�Ӧ��yֵ��ǰ�������ʱ����isSaveFitYsΪtrue
		///
		void getFitedYs(std::vector<double>& fitedYs) { fitedYs = this->fitedYs; }

		/// 
		/// \brief ����x��ȡ��Ϸ��̵�yֵ
		/// \return ����x��Ӧ��yֵ
		///
		template<typename T>
		double getY(const T x) const
		{
			double ans(0);
			for (size_t i = 0; i<factor.size(); ++i)
			{
				ans += factor[i] * pow((double)x, (int)i);
			}
			return ans;
		}
		/// 
		/// \brief ��ȡб��
		/// \return б��ֵ
		///
		double getSlope() { return factor[1]; }
		/// 
		/// \brief ��ȡ�ؾ�
		/// \return �ؾ�ֵ
		///
		double getIntercept() { return factor[0]; }
		/// 
		/// \brief ʣ��ƽ����
		/// \return ʣ��ƽ����
		///
		double getSSE() { return sse; }
		/// 
		/// \brief �ع�ƽ����
		/// \return �ع�ƽ����
		///
		double getSSR() { return ssr; }
		/// 
		/// \brief ���������
		/// \return ���������
		///
		double getRMSE() { return rmse; }
		/// 
		/// \brief ȷ��ϵ����ϵ����0~1֮����������������ж�����Ŷȵ�һ����
		/// \return ȷ��ϵ��
		///
		double getR_square() { return 1 - (sse / (ssr + sse)); }
		/// 
		/// \brief ��ȡ����vector�İ�ȫsize
		/// \return ��С��һ������
		///
		template<typename T>
		size_t getSeriesLength(const std::vector<typename T>& x
			, const std::vector<typename T>& y)
		{
			return (x.size() > y.size() ? y.size() : x.size());
		}
		/// 
		/// \brief �����ֵ
		/// \return ��ֵ
		///
		template <typename T>
		static T Mean(const std::vector<T>& v)
		{
			return Mean(&v[0], v.size());
		}
		template <typename T>
		static T Mean(const T* v, size_t length)
		{
			T total(0);
			for (size_t i = 0; i<length; ++i)
			{
				total += v[i];
			}
			return (total / length);
		}
		/// 
		/// \brief ��ȡ��Ϸ���ϵ���ĸ���
		/// \return ��Ϸ���ϵ���ĸ���
		///
		size_t getFactorSize() { return factor.size(); }
		/// 
		/// \brief ���ݽ״λ�ȡ��Ϸ��̵�ϵ����
		/// ��getFactor(2),���ǻ�ȡy=a0+a1*x+a2*x^2+����+apoly_n*x^poly_n��a2��ֵ
		/// \return ��Ϸ��̵�ϵ��
		///
		double getFactor(size_t i) { return factor.at(i); }
	private:
		template<typename T>
		void calcError(const T* x
			, const T* y
			, size_t length
			, double& r_ssr
			, double& r_sse
			, double& r_rmse
			, bool isSaveFitYs = true
		)
		{
			T mean_y = Mean<T>(y, length);
			T yi(0);
			fitedYs.reserve(length);
			for (int i = 0; i<length; ++i)
			{
				yi = getY(x[i]);
				r_ssr += ((yi - mean_y)*(yi - mean_y));//����ع�ƽ����
				r_sse += ((yi - y[i])*(yi - y[i]));//�в�ƽ����
				if (isSaveFitYs)
				{
					fitedYs.push_back(double(yi));
				}
			}
			r_rmse = sqrt(r_sse / (double(length)));
		}
		template<typename T>
		void gauss_solve(int n
			, std::vector<typename T>& A
			, std::vector<typename T>& x
			, std::vector<typename T>& b)
		{
			gauss_solve(n, &A[0], &x[0], &b[0]);
		}
		template<typename T>
		void gauss_solve(int n
			, T* A
			, T* x
			, T* b)
		{
			int i, j, k, r;
			double max;
			for (k = 0; k<n - 1; k++)
			{
				max = fabs(A[k*n + k]); /*find maxmum*/
				r = k;
				for (i = k + 1; i<n - 1; i++) {
					if (max<fabs(A[i*n + i]))
					{
						max = fabs(A[i*n + i]);
						r = i;
					}
				}
				if (r != k) {
					for (i = 0; i<n; i++)         /*change array:A[k]&A[r] */
					{
						max = A[k*n + i];
						A[k*n + i] = A[r*n + i];
						A[r*n + i] = max;
					}
				}
				max = b[k];                    /*change array:b[k]&b[r]     */
				b[k] = b[r];
				b[r] = max;
				for (i = k + 1; i<n; i++)
				{
					for (j = k + 1; j<n; j++)
						A[i*n + j] -= A[i*n + k] * A[k*n + j] / A[k*n + k];
					b[i] -= A[i*n + k] * b[k] / A[k*n + k];
				}
			}

			for (i = n - 1; i >= 0; x[i] /= A[i*n + i], i--)
				for (j = i + 1, x[i] = b[i]; j<n; j++)
					x[i] -= A[i*n + j] * x[j];
		}
	};
}


#endif