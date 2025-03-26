# Define variables
TARGET_DIR=docs
KATEX_HEADER=katex.html
DOCS_DIR=bintensors

# Ensure the target directory exists
$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

# Generate documentation for bintensors with a custom header
docs: $(TARGET_DIR)
	cd $(DOCS_DIR) && \
	RUSTDOCFLAGS="--html-in-header=$(KATEX_HEADER)" cargo doc --no-deps --target-dir=$(TARGET_DIR)

docs-open: $(TARGET_DIR)
	cd $(DOCS_DIR) && \
	RUSTDOCFLAGS="--html-in-header=$(KATEX_HEADER)" cargo doc --no-deps --target-dir=$(TARGET_DIR) --open

# Clean docs
clean:
	rm -rf $(TARGET_DIR)

.PHONY: docs clean
