#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Define the target file using the script directory
CONFIG_FILE="$SCRIPT_DIR/src/config.ts"
SEARCH_PATTERN='base: import.meta.env.PUBLIC_BASE || "/paper_machine"'
REPLACE_PATTERN='base: import.meta.env.PUBLIC_BASE || "/"'

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file not found at $CONFIG_FILE"
  exit 1
fi

# Use sed to replace the line in config.ts
# The -i '' option is for macOS compatibility for in-place editing without backup
# Use a different delimiter for sed like # or | if patterns contain /
sed -i '' "s#${SEARCH_PATTERN},#${REPLACE_PATTERN},#" "$CONFIG_FILE"

# Check if sed command was successful
if [ $? -eq 0 ]; then
  echo "Successfully updated base path in $CONFIG_FILE"
else
  echo "Error updating $CONFIG_FILE. Exiting."
  exit 1
fi

# Change to the website directory (one level up from script dir) before running pnpm
cd "$SCRIPT_DIR"

# Run the build command
echo "Running pnpm run build..."
pnpm run build

# Check if build command was successful
if [ $? -eq 0 ]; then
  echo "Build successful."
else
  echo "Build failed."
  exit 1
fi

echo "Script finished." 