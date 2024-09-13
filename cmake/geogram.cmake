if(TARGET geogram::geogram)
    return()
endif()

include(FetchContent)
set(install_exploragram cd src/lib && git clone git@github.com:BDoignies/exploragram.git)
FetchContent_Declare(
    geogram
    GIT_REPOSITORY git@github.com:BrunoLevy/geogram.git
    GIT_TAG main
    GIT_SHALLOW TRUE
    UPDATE_DISCONNECTED 1 
    PATCH_COMMAND ${install_exploragram}
)
FetchContent_MakeAvailable(geogram)