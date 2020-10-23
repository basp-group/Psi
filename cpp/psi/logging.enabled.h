#ifndef PSI_LOGGING_ENABLED_H
#define PSI_LOGGING_ENABLED_H

#include "psi/config.h"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "psi/exception.h"

namespace psi {
namespace logging {
void set_level(std::string const &level, std::string const &name = "");

//! \brief Initializes a logger.
//! \details Logger only exists as long as return is kept alive.
inline std::shared_ptr<spdlog::logger> initialize(std::string const &name = "") {
  auto const result = spdlog::stdout_logger_mt(default_logger_name() + name);
  set_level(default_logging_level(), name);
  return result;
}

//! Returns shared pointer to logger or null if it does not exist
inline std::shared_ptr<spdlog::logger> get(std::string const &name = "") {
  return spdlog::get(default_logger_name() + name);
}

//! \brief Sets loggin level
//! \details Levels can be one of
//!     - "trace"
//!     - "debug"
//!     - "info"
//!     - "warn"
//!     - "err"
//!     - "critical"
//!     - "off"
inline void set_level(std::string const &level, std::string const &name) {
  auto const logger = get(name);
  if(not logger)
    PSI_THROW("No logger by the name of ") << name << ".\n";
#define PSI_MACRO(LEVEL)                                                                          \
  if(level == #LEVEL)                                                                              \
  logger->set_level(spdlog::level::LEVEL)
  PSI_MACRO(trace);
  else PSI_MACRO(debug);
  else PSI_MACRO(info);
  else PSI_MACRO(warn);
  else PSI_MACRO(err);
  else PSI_MACRO(critical);
  else PSI_MACRO(off);
#undef PSI_MACRO
  else PSI_THROW("Unknown logging level ") << level << "\n";
}

inline bool has_level(std::string const &level, std::string const &name = "") {
  auto const logger = get(name);
  if(not logger)
    return false;

#define PSI_MACRO(LEVEL)                                                                          \
  if(level == #LEVEL)                                                                              \
  return logger->level() >= spdlog::level::LEVEL
  PSI_MACRO(trace);
  else PSI_MACRO(debug);
  else PSI_MACRO(info);
  else PSI_MACRO(warn);
  else PSI_MACRO(err);
  else PSI_MACRO(critical);
  else PSI_MACRO(off);
#undef PSI_MACRO
  else PSI_THROW("Unknown logging level ") << level << "\n";
}
}
}

//! \macro For internal use only
#define PSI_LOG_(NAME, TYPE, ...)                                                                 \
  if(auto psi_logging_##__func__##_##__LINE__ = psi::logging::get(NAME))                         \
  psi_logging_##__func__##_##__LINE__->TYPE(__VA_ARGS__)

#endif
