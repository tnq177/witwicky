#! /bin/bash
get_seeded_random()
{
  openssl enc -aes-256-cfb -pass pass:"42" -nosalt \
    </dev/zero 2>/dev/null
}
shuf --random-source=<(get_seeded_random) $1 > $2
