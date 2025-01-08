#include <iostream>

#include <argparse/argparse.hpp>

#include <vkgs/engine/engine.h>

int main(int argc, char** argv) {
  argparse::ArgumentParser parser("vkgs");
  parser.add_argument("-i", "--input").help("input ply file.");
  parser.add_argument("-t", "--trajectory").help("input trajectory folder.");
  try {
    parser.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  try {
    vkgs::Engine engine;

    if (parser.is_used("input")) {
      auto ply_filepath = parser.get<std::string>("input");
      engine.LoadSplats(ply_filepath);
    }
    if (parser.is_used("trajectory")) {
      auto trajectory_path = parser.get<std::string>("trajectory");
      engine.LoadTrajectory(trajectory_path);
    }
    // engine.addCamera();
    engine.Run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
