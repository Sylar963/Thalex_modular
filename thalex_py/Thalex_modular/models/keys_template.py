"""
Thalex API Keys Configuration Template

Instructions:
1. Rename this file to keys.py
2. Fill in your API keys from Thalex
3. Keep this file secure and never commit it to version control

To get API keys:
1. Go to https://testnet.thalex.com (for testnet) or https://thalex.com (for mainnet)
2. Create an account and complete verification
3. Go to API Keys section
4. Generate new API key pair
5. Copy the key ID and private key here
"""

from thalex import Network

# API Key IDs
key_ids = {
    Network.TEST: "K619413731",
    Network.PROD: "your_mainnet_key_id_here"  # Only add when ready for production
}

# Private Keys
private_keys = {
    Network.TEST:  """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA17z3VVK9Xlwq+2GmOma/c9Fthk/V5I9lL6zdfP3cguEKdjIV
UB+AvbXPUPGpvNV6ci1PplWp4v12WXQ/C+IPZETj3PKL2xy0cM4TrBnmihep5/cQ
aMvv0SCpHtD2Doog7OFSgNqK2TEH76D/Nl8fEVpOYqOtPAfAAnYZI+HKJrY5BQtG
Xv81WUuQOLf8OfNE22rBHoiqpmjDO9W2G/l7k53+XPLUvjF8Qw2XVRU2HIQgd1mc
aVCJBJUW9WYLfrffvMz+0QvAXzK+t+50QHFnuNTfS1qw+NuMtLPIYcJ1EZbA7xl8
qF1mYtNYe9/tnDbdvN+Ghd+C1fE+oLDF3RD81QIDAQABAoIBAC3Nhs1spZvVAaMh
VkNr8RXLzu8HIChIcXkvwE57L0fyM2BovbGnHpQod3198UWQJVD8Yb60zT7OBUR0
s7X4JsBpB9+u9xZr/7s7ZV3SmemToJUJFUjKk33Y608JmlP364mfRP7qZzQ5fq/X
hJesllH+1EmI6oymMJPVMv01QM/62EqgAglDDOfVB/pAe0LVQ81XlWIYd3keRjGd
dmEYMy9yvCO78fbosEzNABJ4pG44z0mi/THRQMxpQvpavmWkHbGMnodF/VccBM7k
rLjlOW+sGnWgPsAWBbObBIF6r/2dzoYiOY64V0o/5OVrSJAFfCbFhMr7/OsDIgtA
2IczffkCgYEA8MSZ7oGlsz8oIKJk1eXAjLZ7jMQqMjrIL6FrsVdvMRrvC6QnkLne
ceID4WC+apgcajY8Bv0c43BU6ll1X0D2LIew3RstqHXGHY+dpPJ0QBzg0f6EVjp9
BYo7NLn8dV7vizgIBDewej8/OpPFJgQzPCf+Vy1m7Qfsi0YC4MTN87sCgYEA5WL9
dLDElfTdHpKGtkGkfEN80xpq8+Ixf2dFoEN7FqFYl5DeAEs4mQP0BV+EVWsQzioc
r2xzeqSDmTSeMiPg6qFvlj/+58wbyXiBwesUCWHI9a5f3BSlJXBMo3gZR//SOTZD
aQqZ5d9e/zruslZwshlR3edN8E7h1X7C35xwIK8CgYEAk/M75CQm/o7Ayc0Aya/1
MoKwSUAB7fvRJ/O5ibCA01wJqM2mwnh1COYqHESmvjnavCm3mVPQfLJ6e8edKHty
yspXFIeu9uXoaCHobYPPi9YzENel2pb2XIElALGJQValPJeh1XWjLHvRDt3fOA08
rqqk0E1GAkHsSWksO5K0PCECgYEAuidAay99CfkCbWoZ+tSAqPuX1DWvMCaTZsIn
Zez9ehsMK0w8bV6eGsdzg9zFJxDRPY49YzuO56uUxINIEoa9Y4wJY5Shx/kDX9f0
7atZwldh38dYMeFrOFvPRiYT1jNMpnNb92XMCRniHRz1UzMFF/OmVX/95xQM/9Z0
TUXGVS8CgYBwaXImL+3s/iAfF8hiTwvfo2wsUhuXn4+Bph9UcOF1zrC3Sq1yEn7p
dEaqvRB8wbmhv3+w4c8uRyCiqqDfktP8QxgXDLGdtsFaLUGzIlRSD+2YrYMzoruZ
Iv8JGOWCHxgXnPGSYChdJdwro0QJpcDOHmZbNauZKSJssydpRLPmNg==
-----END RSA PRIVATE KEY-----
""",
    Network.PROD: """-----BEGIN PRIVATE KEY-----
your_mainnet_private_key_here
-----END PRIVATE KEY-----"""  # Only add when ready for production
} 