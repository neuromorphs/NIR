{
  description = "Neuromorphic Intermediate Representation reference implementation"; 
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system}; 
      in { 
        devShells.default = 
          let pythonPackages = pkgs.python3Packages;
          in pkgs.mkShell rec {
            name = "impurePythonEnv";
            venvDir = "./.venv";
            buildInputs = [
              pythonPackages.python
              pythonPackages.venvShellHook
              pythonPackages.numpy
              pythonPackages.h5py
              pythonPackages.black
              pythonPackages.pytest
              pkgs.ruff
              pkgs.autoPatchelfHook
            ];
            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
            '';
            postShellHook = ''
              unset SOURCE_DATE_EPOCH
            '';
          };
      }
    );
}
