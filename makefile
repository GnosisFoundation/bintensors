# Define variables
ROOT_DIR := $(shell pwd)
TARGET_DIR := $(ROOT_DIR)/../docs
KATEX_HEADER := $(ROOT_DIR)/docs/katex.html
DOCS_DIR := $(ROOT_DIR)/bintensors

# Ensure the target directory exists
$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

# Make sure the KaTeX header exists before trying to use it
$(KATEX_HEADER):
	@echo "Error: KaTeX header file not found at $(KATEX_HEADER)"
	@exit 1

# Base documentation command
define docs_command
	cd $(DOCS_DIR) && \
	RUSTDOCFLAGS="--html-in-header=$(KATEX_HEADER)" cargo doc --no-deps --target-dir=$(TARGET_DIR) $(1)
endef

# Generate documentation for bintensors with a custom header
docs: $(TARGET_DIR) $(KATEX_HEADER)
	$(call docs_command)

# Generate and open documentation
docs-open: $(TARGET_DIR) $(KATEX_HEADER)
	$(call docs_command,--open)

.PHONY: docs docs-open