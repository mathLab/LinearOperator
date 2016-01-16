/*
 * Additional interface required by linear operator to function
 * properly.
 */

size_t memory_consumption() {
  return sizeof(*this);
}
