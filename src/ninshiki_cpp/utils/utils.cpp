#include <vector>
#include <algorithm>
#include <sstream>
#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ninshiki_cpp/utils/utils.hpp"

namespace ninshiki_cpp::utils
{

std::string get_host_name()
{
  char hostname[32];
  if (gethostname(hostname, 32) != 0)
    return "";

  return hostname;
}

std::string get_env(std::string env)
{
  return getenv(env.c_str());
}

bool is_root()
{
  return (getuid() == 0);
}

bool is_directory_exist(std::string path)
{
  struct stat st;
  if (stat(path.c_str(), &st) != 0)
    return false;

  return ((st.st_mode & S_IFMT) == S_IFDIR);
}

bool create_directory(std::string path)
{
  std::replace(path.begin(), path.end(), '/', ' ');

  std::vector<std::string> directories;
  std::stringstream ss(path);

  std::string str;
  while (ss >> str)
  {
    directories.push_back(str);
  }

  ss.str("");
  ss.clear();
  for (std::vector<std::string>::iterator it = directories.begin(); it != directories.end(); it++)
  {
    ss << *it << "/";

    if (!is_directory_exist(ss.str()))
      if (mkdir(ss.str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0)
        return false;
  }

  return true;
}

bool is_file_exist(std::string path)
{
  struct stat st;
  if (stat(path.c_str(), &st) != 0)
    return false;

  return ((st.st_mode & S_IFMT) == S_IFREG);
}

bool create_file(std::string path)
{
  std::size_t slash = path.find_last_of("/");
  if (slash < path.size())
  {
    std::string directory = path.substr(0, (slash == path.size()) ? 0 : slash);

    if (!is_directory_exist(directory))
    {
      if (!create_directory(directory))
        return false;
    }
  }

  std::ofstream output;
  output.open(path);
  if (output.is_open() == false)
    return false;

  output.close();

  return true;
}

}  // namespace ninshiki_cpp::utils
