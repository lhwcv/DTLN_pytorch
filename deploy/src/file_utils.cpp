#include "file_utils.h"
#include <string>
#include <sys/stat.h>

bool file_exist (const std::string& name)
{
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}
