#pragma once

#define desc tensor_descriptor

struct tensor_descriptor
{
    int batch;
    int channels;
    int height;
    int width;
    int size;
};
