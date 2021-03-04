# - Try to find BLACS
# Once done, this will define
#
#  BLACS_FOUND - system has BLACS
#  BLACS_LIBRARIES - libraries to link to

# Use pkg-config to get hints about paths

if(BLACS_LIBRARIES)

  set(text "BLACS: ${BLACS_LIBRARIES}" )
  message_verbose(text)

else()

  if(mkl)
    
    find_library( mkl_blacs_intelmpi_lp64 mkl_blacs_openmpi_lp64 mkl_blacs_sgimpi_lp64 BLACS_LIBRARIES  BLACS_LIBRARY_PATH  ${BLACS_FIND_REQUIRED})

  else()

    find_library( blacsF77init  BLACS01  BLACS_LIBRARY_PATH ${BLACS_FIND_REQUIRED})
    find_library( blacs         BLACS02  BLACS_LIBRARY_PATH ${BLACS_FIND_REQUIRED})
    list(APPEND BLACS_LIBRARIES  ${BLACS02}  ${BLACS01}  ${BLACS02}  )
    set(BLACS_LIBRARIES ${BLACS_LIBRARIES} CACHE FILEPATH "Blacs libraries" FORCE)

  endif()

endif()

# deal with QUIET and REQUIRED argument

include(FindPackageHandleStandardArgs)


