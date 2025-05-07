/*
 * This file is part of the program ltl2dstar (http://www.ltl2dstar.de/).
 * Copyright (C) 2005-2018 Joachim Klein <j.klein@ltl2dstar.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

/** @file
 * The posix version of RunProgram.
 */

#include "common/RunProgram.hpp"

#ifdef RUNPROGRAM_POSIX

#include <unistd.h>
#include <pwd.h>
#include <cerrno>


#include "common/Exceptions.hpp"


std::string expand_path(const std::string& path);

RunProgram::RunProgram(const std::string& program,
		       const std::vector<std::string>& arguments,
		       bool inheritStdStreams,
		       TempFile* std_in,
		       TempFile* std_out,
		       TempFile* std_err) {
  std::string path=expand_path(program);

  int rv;  

  if (path.find("/")==std::string::npos) {
    // path contains no /
    // check if there exists a file in the current directory
    // that can be executed...

    std::string path_cur_dir = "./"+path;
    struct stat buf;
    if (stat(path_cur_dir.c_str(), &buf)==0)
      if (S_ISREG(buf.st_mode))
	  if (access(path_cur_dir.c_str(), X_OK)==0) {
      // yes, use binary in current directory
      path = path_cur_dir;
    }
  }

  const char *filename=path.c_str();

  if (path.find("/")!=std::string::npos) {
    // path contains /, will not be searched for on the PATH
    // check if file exists + if it is a regular file
    struct stat buf;
    if (stat(filename, &buf)!=0) {
      rv=errno;
      THROW_EXCEPTION(Exception, std::string("Trying to execute '")+filename+"' failed: "+strerror(rv));
    }

    if (!S_ISREG(buf.st_mode)) {
      THROW_EXCEPTION(Exception, std::string("Trying to execute '")+filename+"' failed: Not a regular file");
    }
    
    // check if we have execute permission
    if (access(filename, X_OK)!=0) {
      rv=errno;
      THROW_EXCEPTION(Exception, std::string("Trying to execute '")+filename+"' failed: "+strerror(rv));
    }
  }

  char ** argv=new char *[arguments.size()+2];
  
  for (unsigned int i=0;i<arguments.size();i++) {
    argv[i+1]=(char*)arguments[i].c_str();
    _invocation_details+=" '"+arguments[i]+"'";
  }

  argv[0]=(char*)filename;
  argv[arguments.size()+1]=0;
  _invocation_details=std::string(filename) + " " + _invocation_details;

  int fd_stdin=-1, fd_stdout=-1, fd_stderr=-1;
  bool close_stdin=false, close_stdout=false, close_stderr=false;
  
  if (std_in!=0) {
    std_in->reset();
    fd_stdin=std_in->getFileDescriptor();
  } else {
    if (!inheritStdStreams) {    
      fd_stdin=open("/dev/null", O_RDONLY);
      close_stdin=true;
    }
  }
  
  if (std_out!=0) {
    std_out->reset();
    fd_stdout=std_out->getFileDescriptor();
  } else {
    if (!inheritStdStreams) {
      fd_stdout=open("/dev/null", O_WRONLY);
      close_stdout=true;
    }
  }
    
  if (std_err!=0) {
    std_err->reset();
    fd_stderr=std_err->getFileDescriptor();
  } else {
    if (!inheritStdStreams) {
      fd_stderr=open("/dev/null", O_WRONLY);
      close_stderr=true;
    }
  }
  
  _id=fork();
  if (_id == 0) {
    // child
    if (!inheritStdStreams) {
      // close the standard streams
      close(0);
      close(1);
      close(2);
    }

    if (fd_stdin!=-1) {
      if (inheritStdStreams) {
	close(0);
      }
      dup2(fd_stdin, 0);
      lseek(0, 0, SEEK_SET);
      close(fd_stdin);
    }
    if (fd_stdout!=-1) {
      if (inheritStdStreams) {
	close(1);
      }
      dup2(fd_stdout, 1);
      lseek(1, 0, SEEK_SET);
      close(fd_stdout);
    }
    if (fd_stderr!=-1) {
      if (inheritStdStreams) {
	close(2);
      }
      dup2(fd_stderr, 2);
      lseek(2, 0, SEEK_SET);
      close(fd_stderr);
    }

    execvp(filename, argv); // should not return
    int rv=errno;      
    
    // try to write reason on stderr
    FILE *err=fdopen(2, "w");
    fprintf(err, "execvp failed: %s\n", strerror(rv));
    fprintf(err, "  for program: %s\n", filename);
    fprintf(err, "and arguments:\n");
    while (argv!=0) {
      fprintf(err, "   %s\n", *argv);
      ++argv;
    }
    _exit(1);
  } else if (_id == -1) {
    // fork failed
    THROW_EXCEPTION(Exception, "fork failed...");
  }

  if (close_stdin) {
    close(fd_stdin);
  }

  if (close_stdout) {
    close(fd_stdout);
  }

  if (close_stderr) {
    close(fd_stderr);
  }

  // parent
  delete[] argv;
}

RunProgram::~RunProgram() {
}

int 
RunProgram::waitForTermination() {
  processID_t pid;
  int status;

 wait_again:
  pid=waitpid(_id, &status, 0);

  if (pid==-1) {
    // waitpid error
    switch (errno) {
    case EINTR:
      goto wait_again;
    case ECHILD:
      THROW_EXCEPTION(Exception, "waitpid error: no child to wait for");
    case EFAULT:
    case EINVAL:
      THROW_EXCEPTION(Exception, "waitpid error: "+boost::lexical_cast<std::string>(errno));
    }
  }

  if (pid==_id) {
    if (WIFEXITED(status)) {
      return WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      std::cerr << "Child aborted with SIGNAL "<< WEXITSTATUS(status) << ", "<< WTERMSIG(status) << std::endl;
      return -WTERMSIG(status);
    } else {
      THROW_EXCEPTION(Exception, "Implementation Error: Unkown status of child process");
    }
  } else {
    THROW_EXCEPTION(Exception, "waitpid error: PID mismatch");
  }
}



/** Expand ~ in path */
std::string expand_path(const std::string& path)
{
  if (path.length()==0 || path[0]!='~') {
    // doesn't start with ~
    return path;
  }
  
  const char *tilde_expansion =(char*)0;

  std::string::size_type pos = path.find_first_of('/');    

  if (path.length() == 1 ||   // path == ~
      path[1]=='/') {         // path == ~/...
    tilde_expansion = getenv("HOME");
    if (!tilde_expansion) {
      // HOME is not set in environment, try /etc/passwd
      struct passwd *pw = getpwuid(getuid());
      if (pw) {
	tilde_expansion = pw->pw_dir;
      }
    }
  } else { // path = ~user/
    // extract user
    std::string user(path, 1, 
		     (pos==std::string::npos) ? std::string::npos : pos-1);

    // try to get directory from /etc/passwd
    struct passwd *pw = getpwnam(user.c_str());
    if (pw) {
      tilde_expansion = pw->pw_dir;
    }
  }

  // if we didn't find a tilde expansion, return unchanged path
  if (!tilde_expansion) {
    return path;
  }

  std::string result(tilde_expansion);

  if (pos==std::string::npos) {
    // there is no / in the path, just expand
    return result;
  }

  if (result.length()==0  ||
      result[result.length()-1]!='/') {
    result += '/';
  }

  result+=path.substr(pos+1);

  return result;
}

#endif



