#!/bin/bash

# Generate SSL certificates for Streamlit
SSL_DIR="/app/ssl"
mkdir -p $SSL_DIR

# Generate private key
openssl genrsa -out $SSL_DIR/key.pem 2048

# Generate certificate signing request
openssl req -new -key $SSL_DIR/key.pem -out $SSL_DIR/csr.pem -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Generate self-signed certificate
openssl x509 -req -days 365 -in $SSL_DIR/csr.pem -signkey $SSL_DIR/key.pem -out $SSL_DIR/cert.pem

# Clean up CSR
rm $SSL_DIR/csr.pem

echo "SSL certificates generated in $SSL_DIR"
