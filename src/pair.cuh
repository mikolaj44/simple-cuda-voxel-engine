#pragma once

template<typename A, typename B>
class Pair {
public:
    A first;
    B second;

    __device__ Pair() : first(A{}), second(B{}) {};

    __device__ Pair(const A& first_, const B& second_) : first(first_), second(second_) {};

    friend __device__ bool operator==(const Pair<A, B>& p1, const Pair<A, B>& p2){
        return p1.first == p2.first && p1.second == p2.second;
    }

    friend __device__ bool operator!=(const Pair<A, B>& p1, const Pair<A, B>& p2){
        return !(p1 == p2);
    }
};

template<typename A, typename B, typename C>
class Triple {
public:
    A first;
    B second;
    C third;

    __device__ Triple() : first(A{}), second(B{}), third(C{}) {};

    __device__ Triple(const A& first_, const B& second_, const C& third_) : first(first_), second(second_), third(third_) {};

    friend __device__ bool operator==(const Triple<A, B, C>& t1, const Triple<A, B, C>& t2){
        return t1.first == t2.first && t1.second == t2.second && t1.third == t2.third;
    }

    friend __device__ bool operator!=(const Triple<A, B, C>& t1, const Triple<A, B, C>& t2){
        return !(t1 == t2);
    }
};