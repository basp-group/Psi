#ifndef PSI_LOGGING_H
#define PSI_LOGGING_H

#include "psi/config.h"

#ifdef PSI_DO_LOGGING
#include "psi/logging.enabled.h"
#else
#include "psi/logging.disabled.h"
#endif

//! \macro Normal but significant condition or critical error
#define PSI_NOTICE(...) PSI_LOG_(, critical, __VA_ARGS__)
//! \macro Something is definitely wrong, algorithm exits
#define PSI_ERROR(...) PSI_LOG_(, error, __VA_ARGS__)
//! \macro Something might be going wrong
#define PSI_WARN(...) PSI_LOG_(, warn, __VA_ARGS__)
//! \macro Verbose informational message about normal condition
#define PSI_INFO(...) PSI_LOG_(, info, __VA_ARGS__)
//! \macro Output some debugging
#define PSI_DEBUG(...) PSI_LOG_(, debug, __VA_ARGS__)
//! \macro Output internal values of no interest to anyone
//! \details Except maybe when debugging.
#define PSI_TRACE(...) PSI_LOG_(, trace, __VA_ARGS__)

//! High priority message
#define PSI_HIGH_LOG(...) PSI_LOG_(, critical, __VA_ARGS__)
//! Medium priority message
#define PSI_MEDIUM_LOG(...) PSI_LOG_(, info, __VA_ARGS__)
//! Low priority message
#define PSI_LOW_LOG(...) PSI_LOG_(, debug, __VA_ARGS__)
#endif
