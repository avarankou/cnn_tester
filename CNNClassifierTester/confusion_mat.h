#pragma once

#include <cassert>
#include <vector>

namespace clf
{
    class vector_counter : public std::vector<size_t>
    {
    public:
        vector_counter() {};

        vector_counter(const size_t size, const size_t value = 0)
            : std::vector<size_t>(size, value)
            , _Size(size) {}

        virtual ~vector_counter() = default;

        vector_counter & operator += (const vector_counter & right)
        {
            assert(_Size == right._Size);

            for (size_t i = 0; i < _Size; ++i)
                this->data()[i] += right.data()[i];

            return *this;
        }

    private:
        size_t _Size;
    };

    class confusion_matrix
    {
    public:
        confusion_matrix() {};

        confusion_matrix(const size_t size, const size_t value = 0)
            : mat(size + 1, vector_counter(size + 1, value))
            , _Size(size) {};
        ~confusion_matrix() = default;

        inline void add(size_t x, size_t y)
        {
            ++mat[x][y];
            ++mat[x][_Size];
            ++mat[_Size][y];
            ++mat[_Size][_Size];
        }

        inline size_t operator() (size_t x, size_t y) const
        {
            return mat[x][y];
        }

        inline size_t get(size_t x, size_t y) const
        {
            return mat[x][y];
        }

        inline double getAccuracy(size_t x, size_t y) const
        {
            return mat[x][_Size] ? 1. * mat[x][y] / mat[x][_Size] : 0.;
        }

        inline size_t getAccuracyPercent(size_t x, size_t y) const
        {
            return (size_t)(100. * getAccuracy(x, y));
        }

        inline size_t getColSum (size_t y) const
        {
            return mat[_Size][y];
        }

        inline size_t getCorrect() const
        {
            size_t correct = 0;
            for (size_t i = 0; i < _Size; ++i)
                correct += mat[i][i];

            return correct;
        }

        inline size_t getRowSum(size_t x) const
        {
            return mat[x][_Size];
        }

        inline size_t size() const
        {
            return mat[_Size][_Size];
        }

        confusion_matrix & operator += (const confusion_matrix & right)
        {
            assert(_Size == right._Size);

            for (size_t i = 0; i <= _Size; ++i)
                for (size_t j = 0; j <= _Size; ++j)
                    mat[i][j] += right.mat[i][j];

            return *this;
        }

    private:
        // square confusion matrix _Size X _Size
        // mat[i][_Size] = SUM(mat[i][j]) any j from 0 to _Size - 1
        // mat[_Size][j] = SUM(mat[i][j]) any i from 0 to _Size - 1
        // mat[_Size][_Size] = SUM(mat[i][j]) any i, j from 0 to _Size - 1
        std::vector<vector_counter> mat;
        size_t _Size;
    };
}