#ifndef RAY_TRACER_APPLICATION_H
#define RAY_TRACER_APPLICATION_H

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include "frontends/frontend_controller/FrontendController.h"
#include "ApplicationOptions.h"

namespace po = boost::program_options;

class Application {
private:
    int argc;
    char **argv;

    ApplicationOptions options;

    void parseCommandLine();

    void rewriteOptions(boost::program_options::variables_map const &vm);

    po::options_description createGeneralOptions();

    po::options_description createRenderingOptions();

    po::options_description createFrontendsOptions();

    void run();

    std::vector<std::function<Frontend *()>> createFrontendList() const;

public:

    Application(int argc, char *argv[]);
};

#endif //RAY_TRACER_APPLICATION_H
